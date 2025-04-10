import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import csv
from torch.utils.data import ConcatDataset

from model import AVSegmentationModel
from loss import CombinedLoss
import config as config
import warnings
warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`")


def iou_score(y_pred, y_true, smooth=1e-6):
    """Calculate IoU score"""
    y_pred = y_pred.view(-1) > 0.5  # Ensure binary
    y_true = (y_true.view(-1) > 0.5).float()  # Ensure binary

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection

    return (intersection + smooth) / (union + smooth)


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, artery_dir, vein_dir, vessel_dir=None, transform=None, target_type="artery"):
        """
        Dataset for separate artery and vein segmentation

        Args:
            img_dir: Directory with input images
            artery_dir: Directory with artery masks
            vein_dir: Directory with vein masks
            vessel_dir: Optional directory with vessel masks (combined artery+vein)
            transform: Albumentations transformations
            target_type: Which mask to use as target ("artery", "vein", or "vessel")
        """
        self.img_dir = img_dir
        self.artery_dir = artery_dir
        self.vein_dir = vein_dir
        self.vessel_dir = vessel_dir
        self.transform = transform
        self.target_type = target_type

        self.images = sorted([f for f in os.listdir(
            img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        if artery_dir and os.path.exists(artery_dir):
            self.artery_masks = sorted([f for f in os.listdir(
                artery_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            self.artery_masks = None

        if vein_dir and os.path.exists(vein_dir):
            self.vein_masks = sorted([f for f in os.listdir(
                vein_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            self.vein_masks = None

        if vessel_dir and os.path.exists(vessel_dir):
            self.vessel_masks = sorted([f for f in os.listdir(
                vessel_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            self.vessel_masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read target mask based on target_type
        if self.target_type == "artery" and self.artery_masks:
            mask_path = os.path.join(self.artery_dir, self.artery_masks[idx])
        elif self.target_type == "vein" and self.vein_masks:
            mask_path = os.path.join(self.vein_dir, self.vein_masks[idx])
        elif self.target_type == "vessel" and self.vessel_masks:
            mask_path = os.path.join(self.vessel_dir, self.vessel_masks[idx])
        else:
            raise ValueError(
                f"Invalid target_type: {self.target_type} or directory not found")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to torch tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask


def get_transforms(train=False):
    """Get data transformations for training and validation"""
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-30, 30), scale=(0.8, 1.2),
                     translate_percent=(-0.1, 0.1), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return transform


def save_checkpoint(model, optimizer, epoch, metric_value, checkpoint_dir, filename):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_value': metric_value,
    }, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs, checkpoint_dir, start_epoch=0, save_freq=5):
    best_val_loss = float('inf')
    best_iou = 0.0

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare the CSV file for logging metrics
    metrics_csv = os.path.join(checkpoint_dir, "metrics_log.csv")
    with open(metrics_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row. Adjust columns as needed.
        csv_writer.writerow([
            "epoch", "train_loss", "train_iou",
            "val_loss", "val_iou", "val_acc", "val_f1", "val_auc"
        ])

        for epoch in range(start_epoch, num_epochs):
            # ----- Training Phase -----
            model.train()
            train_loss = 0.0
            train_iou = 0.0

            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for images, masks in pbar:
                images = images.to(device)
                masks = (masks.unsqueeze(1) / 255.0).clamp(0, 1).to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_masks = torch.sigmoid(outputs) > 0.5
                    batch_iou = iou_score(pred_masks.float(), masks.float())

                train_loss += loss.item()
                train_iou += batch_iou.item()
                pbar.set_postfix(
                    {"loss": loss.item(), "iou": batch_iou.item()})

            avg_train_loss = train_loss / len(train_loader)
            avg_train_iou = train_iou / len(train_loader)

            # ----- Validation Phase -----
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_f1 = 0.0
            val_acc = 0.0
            val_auc = 0.0

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    images = images.to(device)
                    masks = (masks.unsqueeze(1) / 255.0).clamp(0, 1).to(device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    pred_probs = torch.sigmoid(outputs)
                    pred_masks = pred_probs > 0.5
                    batch_iou = iou_score(pred_masks.float(), masks.float())

                    pred_flat = pred_masks.cpu().numpy().flatten().astype(np.uint8)
                    masks_flat = masks.cpu().numpy().flatten().astype(np.uint8)
                    pred_probs_flat = pred_probs.cpu().numpy().flatten()

                    batch_acc = accuracy_score(masks_flat, pred_flat)
                    batch_f1 = f1_score(masks_flat, pred_flat,
                                        average='weighted', zero_division=0)
                    try:
                        batch_auc = roc_auc_score(masks_flat, pred_probs_flat)
                    except ValueError:
                        batch_auc = 0.0

                    val_loss += loss.item()
                    val_iou += batch_iou.item()
                    val_f1 += batch_f1
                    val_acc += batch_acc
                    val_auc += batch_auc

            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            avg_val_f1 = val_f1 / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
            avg_val_auc = val_auc / len(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(
                f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
            print(
                f"Val Acc: {avg_val_acc:.4f}, Val F1: {avg_val_f1:.4f}, Val AUC: {avg_val_auc:.4f}")

            # Write the metrics for this epoch to the CSV file
            csv_writer.writerow([
                epoch+1, avg_train_loss, avg_train_iou,
                avg_val_loss, avg_val_iou, avg_val_acc, avg_val_f1, avg_val_auc
            ])
            # Flush to ensure the row is written to disk immediately
            csvfile.flush()

            # Save best models based on validation performance
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, avg_val_iou,
                                checkpoint_dir, 'best_model_loss.pth')

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                save_checkpoint(model, optimizer, epoch, avg_val_iou,
                                checkpoint_dir, 'best_model_iou.pth')

            # Save periodic checkpoint (with desired frequency)
            if (epoch + 1) % save_freq == 0:
                save_checkpoint(model, optimizer, epoch, avg_val_iou,
                                checkpoint_dir, 'checkpoint_epoch_last.pth')
            print("******************************************************")

        # Clean-up: If training finishes completely, remove periodic checkpoint files (optional)
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint_epoch_') and file not in ['best_model_loss.pth', 'best_model_iou.pth']:
                os.remove(os.path.join(checkpoint_dir, file))
                print(f"Deleted checkpoint file: {file}")


def main():
    parser = argparse.ArgumentParser(
        description='AV Segmentation Training/Evaluation/Predict')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict'], default='train',
                        help='Operation mode: train, eval, or predict')
    parser.add_argument('--datasets', nargs='+', type=str,
                        choices=['RITE', 'FIVES', 'RMHAS'], default=['RITE'],
                        help='One or more datasets to use for training (model checkpoint lookup)')
    parser.add_argument('--eval_datasets', nargs='+', type=str,
                        choices=['RITE', 'FIVES', 'RMHAS'],
                        help='One or more datasets to use for evaluation or prediction')
    parser.add_argument('--target', type=str, choices=['artery', 'vein', 'vessel'],
                        default='artery', help='Target vessel type to segment')
    parser.add_argument('--epochs', type=int,
                        default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--encoder', type=str,
                        default='resnet34', help='Encoder model')
    parser.add_argument('--checkpoint_dir', type=str, default='../model',
                        help='Directory to save checkpoints (for train and eval) or load from (for predict)')
    parser.add_argument('--file_name', type=str, default=None,
                        help='Optional: Specific file name for prediction. If not provided, predict on all images in the eval_datasets folder.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset paths (unchanged from before)
    dataset_paths = {
        'RITE':   {'img_dir': config.RITE_IMG,   'artery_dir': config.RITE_ART,
                   'vein_dir': config.RITE_VEIN,  'vessel_dir': config.RITE_VES},
        'FIVES':  {'img_dir': config.FIVES_IMG,  'artery_dir': config.FIVES_ART,
                   'vein_dir': config.FIVES_VEIN, 'vessel_dir': config.FIVES_VES},
        'RMHAS':  {'img_dir': config.RMHAS_IMG,  'artery_dir': config.RMHAS_ART,
                   'vein_dir': config.RMHAS_VEIN, 'vessel_dir': config.RMHAS_VES},
    }

    # Build the checkpoint directory name; note this works even if multiple datasets are given.
    model_name = "_".join(args.datasets) + "_" + args.target
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)

    if args.mode == 'train':
        # ----- Training Mode -----
        datasets_list = []
        for d in args.datasets:
            paths = dataset_paths[d]
            ds = SegmentationDataset(
                img_dir=paths['img_dir'],
                artery_dir=paths['artery_dir'],
                vein_dir=paths['vein_dir'],
                vessel_dir=paths['vessel_dir'],
                transform=get_transforms(train=True),
                target_type=args.target
            )
            datasets_list.append(ds)

        # Combine datasets if more than one is provided
        if len(datasets_list) > 1:
            from torch.utils.data import ConcatDataset
            combined_dataset = ConcatDataset(datasets_list)
        else:
            combined_dataset = datasets_list[0]

        # Split into training and validation sets
        val_size = int(len(combined_dataset) * config.VALIDATION_RATIO)
        train_size = len(combined_dataset) - val_size
        train_dataset, val_dataset = random_split(
            combined_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        # Create model, loss function, and optimizer
        model = AVSegmentationModel(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1  # Binary segmentation for single vessel type
        ).to(device)
        criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Resume training if a checkpoint exists
        resume_checkpoint = os.path.join(
            checkpoint_dir, "checkpoint_epoch_last.pth")
        start_epoch = 0
        if os.path.exists(resume_checkpoint):
            print("Resuming training from checkpoint...")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")

        # Start training
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch  # start from the resumed epoch, if any
        )

    elif args.mode == 'eval':
        # ----- Evaluation Mode -----
        if not args.eval_datasets:
            raise ValueError(
                "In eval mode, the --eval_datasets argument must be provided.")

        # Prepare evaluation datasets based on the eval_datasets argument.
        eval_datasets = []
        for d in args.eval_datasets:
            paths = dataset_paths[d]
            ds = SegmentationDataset(
                img_dir=paths['img_dir'],
                artery_dir=paths['artery_dir'],
                vein_dir=paths['vein_dir'],
                vessel_dir=paths['vessel_dir'],
                transform=get_transforms(train=False),
                target_type=args.target
            )
            eval_datasets.append(ds)

        if len(eval_datasets) > 1:

            combined_eval_dataset = ConcatDataset(eval_datasets)
        else:
            combined_eval_dataset = eval_datasets[0]

        eval_loader = DataLoader(combined_eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        # Create the model and load the best checkpoint for evaluation
        model = AVSegmentationModel(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model_iou.pth")
        if not os.path.exists(checkpoint_path):
            print(
                f"Checkpoint file not found at {checkpoint_path} for evaluation!")
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)

        # Evaluate over the evaluation dataset(s)
        eval_loss = 0.0
        eval_iou = 0.0
        eval_f1 = 0.0
        eval_acc = 0.0
        eval_auc = 0.0

        with torch.no_grad():
            for images, masks in tqdm(eval_loader, desc="Evaluation"):
                images = images.to(device)
                masks = (masks.unsqueeze(1) / 255.0).clamp(0, 1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                pred_probs = torch.sigmoid(outputs)
                pred_masks = pred_probs > 0.5
                batch_iou = iou_score(pred_masks.float(), masks.float())

                pred_flat = pred_masks.cpu().numpy().flatten().astype(np.uint8)
                masks_flat = masks.cpu().numpy().flatten().astype(np.uint8)

                pred_probs_flat = pred_probs.cpu().numpy().flatten()

                batch_acc = accuracy_score(masks_flat, pred_flat)
                batch_f1 = f1_score(masks_flat, pred_flat,
                                    average='weighted', zero_division=0)
                try:
                    batch_auc = roc_auc_score(masks_flat, pred_probs_flat)
                except ValueError:
                    batch_auc = 0.0

                eval_loss += loss.item()
                eval_iou += batch_iou.item()
                eval_acc += batch_acc
                eval_f1 += batch_f1
                eval_auc += batch_auc

            num_batches = len(eval_loader)
            print("Evaluation Results:")
            print(f"Loss: {eval_loss / num_batches:.4f}")
            print(f"IoU: {eval_iou / num_batches:.4f}")
            print(f"Accuracy: {eval_acc / num_batches:.4f}")
            print(f"F1 Score: {eval_f1 / num_batches:.4f}")
            print(f"AUC: {eval_auc / num_batches:.4f}")

    elif args.mode == 'predict':
        # ----- Prediction Mode -----
        if not args.eval_datasets:
            raise ValueError(
                "For predict mode, the --eval_datasets argument must be provided.")

        # Load model from checkpoint (model is saved using training dataset name and target)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model_iou.pth")
        if not os.path.exists(checkpoint_path):
            print(
                f"Checkpoint file not found at {checkpoint_path} for prediction!")
            return

        model = AVSegmentationModel(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Use the non-augmented transformation for prediction.
        transform = get_transforms(train=False)

        # Define output directory for predictions.
        out_dir = os.path.join("../saved_image", model_name)
        os.makedirs(out_dir, exist_ok=True)

        # Collect files to process from each evaluation dataset.
        files_to_process = []
        for d in args.eval_datasets:
            # Images are expected in: processed_data/<eval_dataset>/<target>/
            input_dir = os.path.join("../processed_data", d, args.target)
            if args.file_name:
                file_path = os.path.join(input_dir, args.file_name)
                if os.path.exists(file_path):
                    files_to_process.append((file_path, args.file_name))
                else:
                    print(f"File {args.file_name} not found in {input_dir}")
            else:
                for f in os.listdir(input_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(input_dir, f)
                        files_to_process.append((file_path, f))

        # Predict and save output for each file.
        with torch.no_grad():
            for file_path, filename in files_to_process:
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Warning: could not read image {file_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = transform(image=image)
                image_tensor = transformed["image"].unsqueeze(0).to(device)
                output = model(image_tensor)
                pred_probs = torch.sigmoid(output)
                pred_mask = (pred_probs > 0.3).float()

                # Remove extra dimensions and convert to uint8 image.
                pred_mask = pred_mask.squeeze().cpu().numpy()
                pred_mask_img = (pred_mask * 255).astype(np.uint8)
                save_path = os.path.join(out_dir, filename)
                cv2.imwrite(save_path, pred_mask_img)
                print(f"Saved prediction for {filename} at {save_path}")


if __name__ == "__main__":
    main()
