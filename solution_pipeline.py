import os, re, gc, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

SEED = 42
NUM_CLASSES = 56
MODEL_NAME = "google/muril-base-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "/kaggle/input/competitions/anveshan-data-sprint-2-nlp/train.csv"
TEST_PATH  = "/kaggle/input/competitions/anveshan-data-sprint-2-nlp/test.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def has_deva(text):
    return bool(re.search(r'[\u0900-\u097F]', str(text)))

def to_canonical_deva(text):
    text = str(text).strip()
    return text if has_deva(text) else transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)

def norm_logits(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

class TextDataset(Dataset):
    def __init__(self, texts, tok, labels=None, max_len=64):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {"input_ids": self.enc["input_ids"][idx], "attention_mask": self.enc["attention_mask"][idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item

def make_collate(pad_id):
    def collate(batch):
        max_b = max(len(x["input_ids"]) for x in batch)
        out = {
            "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (max_b - len(x["input_ids"])) for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] + [0] * (max_b - len(x["attention_mask"])) for x in batch], dtype=torch.long),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return out
    return collate

class CrossScriptMuRIL(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.attn = nn.Linear(hidden, 1)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        mask = attention_mask.unsqueeze(-1).float()
        seq_f = seq.float()
        mean_pool = (seq_f * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        scores = self.attn(seq).float().masked_fill(mask == 0, -1e9)
        attn_pool = (seq_f * torch.softmax(scores, dim=1)).sum(dim=1)
        feat = torch.cat([mean_pool, attn_pool], dim=-1)
        if self.training:
            return sum(self.fc(dp(feat)) for dp in self.dropouts) / len(self.dropouts)
        return self.fc(feat)

@torch.inference_mode()
def predict_logits(model, loader):
    model.eval()
    preds = []
    for b in loader:
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            preds.append(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)).float().cpu().numpy())
    return np.vstack(preds)

def train_muril_cv(train_raw, train_deva, y_true, test_raw, test_deva, tok, collate, skf, pseudo_df=None):
    oof_muril  = np.zeros((len(train_raw), NUM_CLASSES), dtype=np.float32)
    test_muril = np.zeros((len(test_raw), NUM_CLASSES), dtype=np.float32)
    soup_states = []

    test_loader_orig = DataLoader(TextDataset(test_raw, tok), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader_deva = DataLoader(TextDataset(test_deva, tok), batch_size=64, shuffle=False, collate_fn=collate)

    for fold, (tr_i, va_i) in enumerate(skf.split(train_raw, y_true)):
        tr_texts = [train_raw[i] for i in tr_i]
        tr_y = list(y_true[tr_i])

        if pseudo_df is not None and len(pseudo_df) > 0:
            tr_texts.extend(pseudo_df["query_text"].tolist())
            tr_y.extend(pseudo_df["target"].tolist())

        counts = np.bincount(tr_y, minlength=NUM_CLASSES)
        prior_adj = torch.tensor(0.25 * np.log(counts / counts.sum() + 1e-12), dtype=torch.float32, device=DEVICE)

        tr_loader = DataLoader(TextDataset(tr_texts, tok, tr_y), batch_size=16, shuffle=True, collate_fn=collate)
        va_loader_orig = DataLoader(TextDataset([train_raw[i] for i in va_i], tok), batch_size=64, shuffle=False, collate_fn=collate)
        va_loader_deva = DataLoader(TextDataset([train_deva[i] for i in va_i], tok), batch_size=64, shuffle=False, collate_fn=collate)

        model = CrossScriptMuRIL(MODEL_NAME, NUM_CLASSES).to(DEVICE)
        if fold == 0 and pseudo_df is None:
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params <= 300_000_000, f"Parameter cap exceeded: {n_params}"
            print(f"Verified Neural Parameters: {n_params:,} (<= 300,000,000)")

        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 2.5e-5},
            {"params": list(model.attn.parameters()) + list(model.fc.parameters()), "lr": 1e-4},
        ], weight_decay=0.01)
        epochs = 4
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * len(tr_loader) * epochs), len(tr_loader) * epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

        best_f1, best_sd = -1.0, None
        for _ in range(epochs):
            model.train()
            for b in tr_loader:
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
                    loss = F.cross_entropy(logits + prior_adj, b["labels"].to(DEVICE), label_smoothing=0.05)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

            va_log = 0.65 * predict_logits(model, va_loader_orig) + 0.35 * predict_logits(model, va_loader_deva)
            sc = f1_score(y_true[va_i], va_log.argmax(-1), average="macro")
            if sc > best_f1:
                best_f1, best_sd = sc, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

        soup_states.append(best_sd)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_sd.items()})
        oof_muril[va_i] = 0.65 * predict_logits(model, va_loader_orig) + 0.35 * predict_logits(model, va_loader_deva)
        test_muril += (0.65 * predict_logits(model, test_loader_orig) + 0.35 * predict_logits(model, test_loader_deva)) / 5.0
        del model, opt, sched, scaler
        gc.collect()
        torch.cuda.empty_cache()

    return oof_muril, test_muril, soup_states

def main():
    seed_everything(SEED)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    y_true   = train_df["target"].values

    train_raw  = train_df["query_text"].astype(str).tolist()
    test_raw   = test_df["query_text"].astype(str).tolist()
    train_deva = [to_canonical_deva(t) for t in train_raw]
    test_deva  = [to_canonical_deva(t) for t in test_raw]

    # 1. Cross-Script TF-IDF + Logistic Regression Branch (0M Neural Params)
    train_comb = pd.Series(train_raw) + " || " + pd.Series(train_deva)
    test_comb  = pd.Series(test_raw)  + " || " + pd.Series(test_deva)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_tfidf  = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
    test_tfidf = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

    for tr_i, va_i in skf.split(train_df, y_true):
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=45000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=75000, sublinear_tf=True)
        X_tr = hstack([vw.fit_transform(train_comb.iloc[tr_i]), vc.fit_transform(train_comb.iloc[tr_i])])
        X_va = hstack([vw.transform(train_comb.iloc[va_i]), vc.transform(train_comb.iloc[va_i])])
        X_te = hstack([vw.transform(test_comb), vc.transform(test_comb)])
        clf = LogisticRegression(C=3.0, max_iter=350).fit(X_tr, y_true[tr_i])
        oof_tfidf[va_i] = clf.decision_function(X_va)
        test_tfidf += clf.decision_function(X_te) / 5.0

    # 2. Stage-1 & Stage-2 Fine-Tuned MuRIL with Dual-Script TTA
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate(tok.pad_token_id or 0)

    oof_s1, test_s1, _ = train_muril_cv(train_raw, train_deva, y_true, test_raw, test_deva, tok, collate, skf)
    blend_s1_test = 0.78 * norm_logits(test_s1) + 0.22 * norm_logits(test_tfidf)
    probs_s1 = softmax(blend_s1_test * 1.5, axis=1)
    mask = probs_s1.max(axis=1) >= 0.88
    pseudo_df = pd.DataFrame({
        "query_text": np.array(test_raw)[mask],
        "target": blend_s1_test.argmax(-1)[mask],
    })

    oof_muril, test_muril, soup_states = train_muril_cv(
        train_raw, train_deva, y_true, test_raw, test_deva, tok, collate, skf, pseudo_df=pseudo_df
    )
    torch.save({k: sum(sd[k] for sd in soup_states) / len(soup_states) for k in soup_states[0]}, "prod_muril_final.pt")

    # 3. Z-Score Standardized Logit Blend
    final_oof  = 0.78 * norm_logits(oof_muril)  + 0.22 * norm_logits(oof_tfidf)
    final_test = 0.78 * norm_logits(test_muril) + 0.22 * norm_logits(test_tfidf)
    print(f"5-Fold OOF Macro-F1: {f1_score(y_true, final_oof.argmax(-1), average='macro'):.5f}")

    sub = pd.DataFrame({"example_id": test_df["example_id"], "target": final_test.argmax(-1)})
    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv successfully.")

if __name__ == "__main__":
    main()
