# Prithvi-EO-2.0-300M frozen-encoder track (Gate C candidate)

Kaggle notebook stages — run AFTER the SegFormer baseline is training, not
instead of it. Hard stop: July 22 kill decision by holdout per-class IoU +
visual inspection.

## Stage 1 — environment (verified July 12)
    pip install terratorch
    huggingface-cli login   # token with read access

## Stage 2 — TerraTorch fine-tune, frozen encoder
Use terratorch's semantic segmentation task with:
  - backbone: prithvi_eo_v2_300  (pretrained=True)
  - backbone FROZEN (freeze_backbone: true)
  - decoder: UperNetDecoder (or FCNDecoder for the fastest first signal)
  - bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    -> our chip indices [0, 1, 2, 4, 5, 6]  (see dataset.PRITHVI_BANDS)
  - num_classes: 4, class weights from dataset.class_weights_from_manifest
  - normalization: OUR stats.json values (train-only), NOT terratorch
    defaults — Gate C fairness requires identical normalization across
    backbones. Override the datamodule means/stds explicitly.
  - crop 224, same flips/rot90 augmentation as SegFormer run

Data plumbing options (pick 1):
  A. terratorch generic datamodule pointed at chips/{split}/images+masks
     (GenericNonGeoSegmentationDataModule handles tif pairs), or
  B. wrap our ChipDataset in a LightningDataModule (20 lines) — preferred,
     guarantees identical sampling to the SegFormer run.

## Stage 3 — evaluate for Gate C
Report per-class IoU + mIoU on the GEOGRAPHIC test split, same script logic
as train_segformer.evaluate. Fill the Gate C table:

| backbone      | pretraining        | bands | mIoU | IoU bg | IoU veg | IoU water | IoU built |
|---------------|--------------------|-------|------|--------|---------|-----------|-----------|
| U-Net (S2)    | ImageNet ResNet50  | ...   |      |        |         |           |           |
| SegFormer-B0  | ImageNet (MiT)     | 7     |      |        |         |           |           |
| Prithvi-300M  | HLS EO (4.2M ts)   | 6     |      |        |         |           |           |

## Stage 4 — export the winner for infer_citywide.py
SegFormer wins -> save_pretrained (script loads it natively).
Prithvi wins   -> torch.jit.trace the full model on a (1,6,224,224) example
                  and save; infer_citywide.py --arch prithvi loads the
                  torchscript. (Trace AFTER eval; verify traced == eager on
                  one batch.)

## Kill criteria (July 22, no renegotiation)
KILL if any of: water OR built IoU dramatically below SegFormer's; visually
incoherent predictions on 3 stratified holdout tiles; or the pipeline still
isn't producing evaluable predictions. "Almost works, two more days" = kill.
