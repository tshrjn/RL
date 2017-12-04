import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model_policy import Net
import data_util

def train_policy(args):
    # Get processed data
    train_data, val_data = data_util.process_data(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print("Train Data Size: ", len(train_data))
    print("Val Data Size: ", len(val_data))

    # Neural Network & optimizer
    model=Net(*train_data.dimensions())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training:")
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            validation_loss += F.mse_loss(output, target, size_average=False).data[0] # sum up batch loss
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}\n'.format(validation_loss))


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation()
        model_file = 'model_' + str(epoch) + '.pth' #  
        torch.save(model.state_dict() , 'models/' + model_file)
        print('\nSaved model')