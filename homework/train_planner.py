import argparse
import torch
from torch.utils.data import DataLoader
from homework.models import load_model, save_model  # Import load_model and save_model from models.py
from homework.datasets.road_dataset import RoadDataset # Assuming you have a dataset class defined for handling data

def train(model_name, transform_pipeline, num_workers, lr, batch_size, num_epoch):
    # Load model
    model = load_model(model_name=model_name)
    model.train()  # Set model to training mode
    
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()  # Assuming we're using MSE loss for waypoint prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load dataset
    train_dataset = RoadDataset(transform=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training loop
    for epoch in range(num_epoch):
        epoch_loss = 0
        for batch in train_loader:
            # Assume batch contains inputs and target waypoints
            inputs, targets = batch["inputs"], batch["targets"]

            # Forward pass
            outputs = model(**inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    save_model(model)
    print("Model saved.")

def main():
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train (e.g., 'mlp_planner')")
    parser.add_argument("--transform_pipeline", type=str, default="state_only", help="Transformation pipeline for data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epoch", type=int, default=40, help="Number of epochs")
    
    args = parser.parse_args()
    train(
        model_name=args.model_name,
        transform_pipeline=args.transform_pipeline,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch
    )

if __name__ == "__main__":
    print("Time to train")
    main()
