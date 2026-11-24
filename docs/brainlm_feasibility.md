# BrainLM feasibility scout (CPU, Linux container)

## Verdict
- The official 111M / 650M HF weights (`vandijklab/brainlm`) could NOT be downloaded: **huggingface.co is blocked by the egress proxy** (`huggingface.co:443 connect_rejected`, HTTP 403 on CONNECT; also hf.co, hf-mirror.com, cdn-lfs.huggingface.co, cas-bridge.xethub.hf.co, download.pytorch.org all fail). No workaround attempted. Repo ids, sizes, file lists are therefore UNVERIFIED (taken only from the GitHub README).
- The GitHub repo (cloneable; github.com + pypi.org are reachable) ships a SMALL dev checkpoint in-tree that loads and runs fine on CPU: `pretrained_models/2023-06-06-22_15_00-checkpoint-1400` (5.05M params). Everything below was measured on THAT checkpoint, not the 111M/650M models.

## Source & license
- Code: github.com/vandijklab/BrainLM, cloned at commit eded39c86c27e03f5ead1d6a14311e92d1305e5e (2025-11-22), at /home/user/brainlm_scout/BrainLM.
- README claims weights at https://huggingface.co/vandijklab/brainlm/ "which includes both 111M and 650M param models" (not verified; README also says finetuned models not yet released; large models "require flash attention", setup.sh does `pip install flash_attn`).
- License: LICENSE file first line: "Attribution-NonCommercial-NoDerivatives 4.0 International" (CC BY-NC-ND 4.0); README badge URL https://creativecommons.org/licenses/by-nc-nd/4.0/. HF repo license text not checked (blocked). Flag: NoDerivatives may matter for a study that edits and redistributes weights; non-commercial only. Needs Caleb's judgement.

## Disk/RAM
- Before: Avail 20G (252G fs). After install: Avail ~11G (venv 5.5G, because PyPI torch on x86 Linux pulls the CUDA 13 wheels + nvidia-* libs; download.pytorch.org CPU index is blocked). If space matters: could strip nvidia-* but untested.
- RAM 15 GB total, 4 cores, no swap. Peak RSS 6.4 GB at batch 8 (batch 1 alone not separately measured).
- /root/.claude/scripts/heavy-check.sh does not exist in this container; ran with 2 threads, CPU only.

## Install recipe
```
python3 -m venv /home/user/brainlm_scout/.venv && . /home/user/brainlm_scout/.venv/bin/activate
pip install torch numpy einops psutil          # torch 2.15.0 (CUDA wheel, runs on CPU)
pip install "transformers==4.46.3"             # NOT latest: transformers 5.22.1 lacks ViTMAEEncoder -> ImportError
git clone --depth 1 https://github.com/vandijklab/BrainLM.git /home/user/brainlm_scout/BrainLM
export HF_HOME=/home/user/brainlm_scout/hf_cache OMP_NUM_THREADS=2
python -I /home/user/brainlm_scout/scout.py
```
Versions: Python 3.13.18, torch 2.15.0, transformers 4.46.3, tokenizers 0.20.3, huggingface_hub 0.36.2, safetensors 0.8.0, numpy 2.5.4, einops 0.8.2, psutil 7.2.2. (Repo pins transformers==4.28.0; 4.46.3 imports and loads fine; 4.28 not tried.) Do not install flash_attn or the repo's requirements.txt (very old pins, PyQt5, etc.); not needed for this checkpoint.

## Architecture & params (shipped checkpoint, config.json)
- Stock `transformers` does NOT load it: custom classes in repo `brainlm_mae/` (BrainLMForPretraining, BrainLMConfig; model_type brainlm_mae) required; no `trust_remote_code` mechanism, import via sys.path.
- Encoder: 4 layers, hidden 256, 4 heads, FFN 1024, gelu, **Nystromformer attention layers** (HF NystromformerLayer, 64 landmarks, conv kernel 65) inside a ViTMAE shell. Decoder: 2 layers, hidden 256, 4 heads, FFN 1024 (+ decoder_pred, pred1/pred2 heads).
- Token scheme: each parcel's time series is cut into patches of 20 timepoints -> token = Linear(20->256)(patch) + Linear(3->256)(xyz) + sinusoidal time pos-enc; tokens = 424 parcels x 10 patches = 4240; random masking (mask_ratio 0.2) keeps 3392; + CLS = 3393 tokens.
- Total params 5,047,724 (nn.Parameters; the two 5000x256 sinusoid `pe` buffers are extra in state_dict). Encoder (`vit.*`) 3,167,248. Decoder+heads 1,880,476.
- By prefix (state_dict numel incl. buffers): vit.embeddings 1,286,656 (1.28M of it the pe buffer); vit.encoder.layer 3,160,080 (790,020/layer); vit.layernorm 512; decoder layers 790,020 each x2; decoder_pred 197,376; decoder_pred1 32,896; decoder_pred2 2,580; decoder_embed 65,792; decoder.pos_embedding 1,280,000 (buffer).
- First keys/shapes: vit.embeddings.cls_token (1,1,256); signal_embedding_projection.weight (256,20); xyz_embedding_projection.weight (256,3); pos_embedding.pe (5000,256); vit.encoder.layer.N.attention.self.{query,key,value}.weight (256,256); ...self.conv.weight (4,1,65,1); attention.output.dense (256,256); attention.output.LayerNorm (256); intermediate.dense (1024,256); output.dense (256,1024); output.LayerNorm (256). Full 40-key listing: run scout.py.
- Checkpoint provenance: trainer_state step 1400, eval_loss 0.868, eval_r2 0.142 -- an under-trained dev run, not the released foundation model; interpretability findings on it should not be generalised.

## Input format
- `model(signal_vectors=[B, 424, 200], xyz_vectors=[B, 424, 3], noise=None, output_hidden_states=True)`. 424 = A424 atlas; xyz = columns 2-4 of `toolkit/atlases/A424_Coordinates.dat` (424 rows: id, x, y, z in MNI-like mm, ranges x -62..64, y -92..66, z -54..70), used raw (unscaled) in the repo notebooks as far as checked.
- This checkpoint: 200 timepoints per window, patch 20 (10 tokens/parcel), mask_ratio 0.2. Forward always masks randomly (pass `noise` tensor [B, 4240] for determinism) and returns a masked-patch MSE loss. Pretraining config in code defaults is 490 timepoints/patch 49/mask 0.5 (UKB); the 111M/650M configs are unverified.
- Normalisation: toolkit produces several variants; train.py default column is `Voxelwise_RobustScaler_Normalized_Recording`; tutorial notebook uses `All_Patient_All_Voxel_Normalized_Recording`. UKB-derived; TR not stated in the code I read (UKB TR 0.735 s; unverified here). Exactly which scaling matches which released weights is unresolved.
- Output: logits=(recon [B,424,10,20], latent [B,3393,256]), mask [B,424,10], hidden_states 5 x [B,3393,256].

## Forward-pass timing (2 threads, CPU, random input)
- Batch 1: 2.92 / 2.45 / 2.44 s (~2.6 s/sample). Batch 8: 23.5 / 22.9 / 21.6 s (~2.8 s/sample). Peak RSS 6.4 GB (whole process, includes batch 8). For the 111M/650M models expect much slower (not measured).

## Hooks
- `register_forward_hook` on `model.vit.encoder.layer[i]` fires for all 4 layers; output is a tuple, `out[0]` shape [1, 3393, 256]. A hook that returns a modified tuple (zeroing layer 1) propagated: hidden_states[2] absmax = 0.0. So activation reading and in-place intervention both work.

## Blockers / notes for an editing harness
- Needs repo custom code (import from clone), pinned-ish transformers (<5; 4.46.3 OK).
- Weights are pickled `pytorch_model.bin` (not safetensors); `torch.load(weights_only=True)` succeeded (plain tensors, 124 keys), `load_state_dict` all keys matched. optimizer.pt/training_args.bin are pickles; not loaded.
- Attention is Nystrom (landmark approximations + depthwise conv), not standard softmax attention: attention-map interpretability needs care; heads' q/k/v are separate Linear layers (easy to edit).
- 4240-token sequences: cost grows with batch; batch 8 uses ~6 GB.
- Official 111M/650M weights unreachable from this container (huggingface.co blocked); they may also need flash-attn at inference per README (the repo has a replace_vitmae_attn_with_flash_attn module; unverified). Ask for the host to be allowlisted or supply the weights.
- License CC BY-NC-ND (see above).
- Scripts: /home/user/brainlm_scout/scout.py. Nothing committed.

## Update 2026-11-09: official vitmae_111M weights (after the network allowlist)

- **Source:** `vandijklab/brainlm`, `vitmae_111M/pytorch_model.bin` (450.5 MB). sha256
  25f5b52178f7409dee4fed2fa3dd18c6692d0764d96996ff0ee4be740520fcd6 matches the HF LFS oid. The repo is
  ungated. HF card license: cc-by-nc-nd-4.0.
- **Architecture:** stock HF `ViTMAEForPreTraining`, not the Nystromformer dev checkpoint.
  - Encoder: 12 layers, hidden 768, 12 heads, MLP 3072.
  - Decoder: 8 layers, 512 wide.
  - Patch 16, image 432x432x3, giving 729 patches; mask ratio 0.75.
  - Params: 112.6M total, 86.2M in the encoder. No flash-attn and no custom code are needed to load
    it.
- **Load:** `transformers` 4.46.3 `ViTMAEForPreTraining(ViTMAEConfig.from_pretrained(dir))` +
  `torch.load(weights_only=True)` gives 0 missing and 0 unexpected keys.
- **Speed (CPU, 2 threads):** batch 4 forward 1.39 s (about 0.35 s per sample after warm-up).
  Peak RSS 1.6 GB.
- **Hooks:** a forward hook on `vit.encoder.layer[5]` returns [B, 183, 768] (182 kept patches + CLS).
- **Input format, per the repo tutorial** (`toolkit/BrainLM_Tutorial.ipynb`):
  - an A424 parcel x time matrix [424, 200], robust-scaled per parcel (median / IQR across
    subjects);
  - placed into the 432x432 frame with padding, as 3 identical channels;
  - the tutorial uses `brainlm_mae.modeling_vit_mae_with_padding.ViTMAEForPreTraining`, which shares
    these weights and handles the padded region.
  - The training TR is about 0.72-0.735 s (UKB/HCP).
- Script: `/home/user/brainlm_scout/load111.py` (outside the repo). Optimizer and other training-state
  files were not downloaded.
