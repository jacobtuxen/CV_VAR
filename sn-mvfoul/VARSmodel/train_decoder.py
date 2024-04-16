import torchvision.transforms as transforms
from dataset import SelfSupervisedDataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
import torch
from utils import batch_tensor, unbatch_tensor
from tqdm import tqdm
import argparse
import gc
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--path", type=str, default="/work3/s194572/SoccerData/mvfouls/", help="Path to the dataset")
parser.add_argument("--start_frame", type=int, default=0, help="Start frame")
parser.add_argument("--end_frame", type=int, default=125, help="End frame")
parser.add_argument("--fps", type=int, default=25, help="Frames per second")
parser.add_argument("--split", type=str, default="Train", help="Train or Test")
parser.add_argument("--num_views", type=int, default=4, help="Number of views")
parser.add_argument("--feat_dim", type=int, default=512, help="Feature dimension")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
parser.add_argument("--type", type=str, default="views", help="Choose views/sync")
parser.add_argument("--model", type=str, default="_", help="Load a model")
args = parser.parse_args()

transformAug = transforms.Compose(
    [
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
    ]
)

transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()

dataset_async_view = SelfSupervisedDataset(
    path=args.path,
    start=args.start_frame,
    end=args.end_frame,
    fps=args.fps,
    split=args.split,
    num_views=args.num_views,
    transform=transformAug,
    transform_model=transforms_model,
    semi_type=args.type
)

data_loader_async = torch.utils.data.DataLoader(dataset=dataset_async_view, batch_size=args.batch_size, shuffle=True, pin_memory=True)

# TRAIN
def train():
  model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT).to(args.device)
  model.fc = torch.nn.Sequential()
  lifting_net = torch.nn.Sequential().to(args.device)
  classifier = torch.nn.Sequential(
    torch.nn.Linear(2*args.feat_dim, 50),
    torch.nn.Linear(50, 1),
  ).to(args.device)

  loss_fn = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': classifier.parameters()}
  ], lr=1e-3)

  loss_list = []

  for epoch in tqdm(range(args.epochs)):
    total_loss = 0.0
    for mvclips, target in data_loader_async:
      mvclips = mvclips.to(args.device)
      target = target.to(args.device)
      
      # Run decoder over each view, and then 
      # concatenate flattened features from all views
      tmp0 = batch_tensor(mvclips, dim=1, squeeze=True)
      tmp1 = model(tmp0)
      aux = unbatch_tensor(tmp1, batch_size=args.batch_size, dim=1, unsqueeze=True)
      aux = lifting_net(aux)
      aux = aux.view(-1, args.feat_dim*2)
      prediction = classifier(aux)

      # Calculate the loss
      loss = loss_fn(prediction, target)
      print(loss)
      total_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      gc.collect()
      torch.cuda.empty_cache()

    avg_loss = total_loss / len(data_loader_async)  # Calculate average loss for the epoch
    tqdm.write(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")  # Display the average loss using tqdm
    loss_list.append(avg_loss)

  if True:
    torch.save(model.state_dict(), f'weights/decoder_{args.type}.pth')
    plt.plot(loss_list)
    plt.savefig(f'plots/loss_train_{args.type}.png')

  #


if __name__ == '__main__':
  if args.split == "Train":
    train()
  