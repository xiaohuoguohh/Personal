import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
#from Lenet5 import Lenet5
from ResNet import ResNet18

def main():
    batchsz = 256
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize(32, 32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize(32, 32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_train = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)


    x, label = iter(cifar_train).next()

    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    #model = Lenet5()
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    print(model)

    for epoch in range(150):

        model.train()

        for batchidx, (x, label) in enumerate(cifar_train):
            # x, label = x.to(divice), label.to(divice)
            # [b, 10]


            # loss:

            logits = model(x)

            loss = criterion(logits, label)
            # loss(tensor scalar)
            # backpropagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
        # test
            total_correct = 0
            total_num = 0
            for  x, label in cifar_train:
                logits = model(x)  # [b, 10]
                pred = logits.argmax(dim = 1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print("accuracy:", acc)



if __name__ == '__main__':
    main()