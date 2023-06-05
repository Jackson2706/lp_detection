from lib import *
from multiboxloss import MultiBoxLoss
from src.dataset import licensePlateDataset, my_collate_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.ssd300_vgg16(pretrained = False)
print(model)

criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)
optimizer = optimizer.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---"*20)
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch+1) % 2 == 0:
                    net.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move to GPU
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                print(targets)
                # init optimizer
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    print(outputs)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward() # calculate gradient
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch+1) + ".pth")


train_dataset = licensePlateDataset(root="data", phase="train")
val_dataset = licensePlateDataset(root="data", phase="valid")
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}
num_epochs = 10

train_model(model, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)