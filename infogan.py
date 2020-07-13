import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import wandb
from torchvision.utils import save_image

wandb.init(job_type = 'train', project = 'Info GAN')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bs = 100

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size =bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return torch.sigmoid(self.fc3(x))
    
class Generator(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, z_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

class Auxilarry_NN(nn.Module):
    def __init__(self, z_dim, h1_dim, h2_dim, output_dim):
        super(Auxilarry_NN, self).__init__()
        self.fc1 = nn.Linear(z_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.softmax(self.fc3(x))

z_dim = 10
n_classes = 10

generator = Generator(z_dim + n_classes, 100,100, 784).to(device)
discriminator = Discriminator(784, 100,50).to(device)
auxilarry_nn = Auxilarry_NN(784, 100,100,n_classes).to(device)


lr = 0.0001

optim_generator = optim.Adam(generator.parameters(), lr=lr)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
optim_aux_aux_nn = optim.Adam(auxilarry_nn.parameters(), lr=lr)
optim_aux_gen = optim.Adam(generator.parameters(), lr=lr)


for epoch in range(200):
    discriminator_loss = 0
    generator_loss = 0
    q_nn_loss = 0

    for batch_id, (image,label) in enumerate(train_loader):


        
        z_generated = torch.from_numpy(np.random.uniform(-1,1,size=[bs,z_dim])).float().to(device)
        c_generated = torch.from_numpy(np.random.multinomial(1, n_classes*[1/n_classes], size=bs)).float().to(device)
        input = torch.cat((z_generated, c_generated), dim=1).to(device)

        sample_generated = generator(input)

        fake_output_discriminator = discriminator(sample_generated)

        #Generator Loss
        gen_loss = -torch.mean(torch.log(fake_output_discriminator + 1e-7))

        generator.train()
        optim_generator.zero_grad()
        gen_loss.backward()
        generator_loss += gen_loss.item()
        optim_generator.step()

        image = image.view(-1,784).to(device)

        z_generated = torch.from_numpy(np.random.uniform(-1,1,size=[bs,z_dim])).float().to(device)
        c_generated = torch.from_numpy(np.random.multinomial(1, n_classes*[1/n_classes], size=bs)).float().to(device)
        input = torch.cat((z_generated, c_generated), dim=1).to(device)
        
        sample_generated = generator(input)
        
        fake_output_discriminator = discriminator(sample_generated)
        real_output_discriminator = discriminator(image)
        
        #Discriminator Loss
        disc_loss = -torch.mean(torch.log(1 - fake_output_discriminator + 1e-7) + torch.log(real_output_discriminator + 1e-7))

        optim_discriminator.zero_grad()
        disc_loss.backward()
        discriminator_loss += disc_loss.item()
        optim_discriminator.step()

        z_generated = torch.from_numpy(np.random.uniform(-1,1,size=[bs,z_dim])).float().to(device)
        c_generated = torch.from_numpy(np.random.multinomial(1, n_classes*[1/n_classes], size=bs)).float().to(device)
        input = torch.cat((z_generated, c_generated), dim=1).to(device)
        
        sample_generated = generator(input)
        estimated_c = auxilarry_nn(sample_generated)
        
        #Auxillary Loss
        q_loss = -torch.mean(torch.sum(torch.log(estimated_c + 1e-7)*c_generated))

        optim_generator.zero_grad()
        optim_aux_aux_nn.zero_grad()
        q_loss.backward()
        q_nn_loss += q_loss.item()
        optim_aux_aux_nn.step()
        optim_aux_gen.step()
    
    wandb.log({'Discriminator Loss': discriminator_loss/len(train_loader.dataset), 'Generator Loss':generator_loss/len(train_loader.dataset), 'Auxilarry Loss':q_nn_loss/len(train_loader.dataset)}, step=epoch)
    print('Epoch {} \t Discriminator_loss {} \t Generator_loss {} \t Q_loss {}'.format(epoch, discriminator_loss/len(train_loader.dataset), generator_loss/len(train_loader.dataset), q_nn_loss/len(train_loader.dataset)))




    with torch.no_grad():
        z = np.random.uniform(-1,1,size = [10,z_dim])
        label = range(10)
        c = np.zeros((10 , n_classes))
        for i, l in enumerate(label):
            c[i,l] = 1
        c = torch.from_numpy(c).float().to(device)
        z = torch.from_numpy(z).float().to(device)
        input = torch.cat((z,c), dim=1)
        out = generator(input)

        sample = out.view(10,1,28,28).to(device)
        wandb.log({"Images": [wandb.Image(sample, caption="Images for epoch: "+str(epoch))]}, step=epoch)
        # if epoch % 10 == 1:
        save_image(sample, './images/sample_'+str(epoch)+'.png')

    

    torch.save(generator.state_dict(), './ckpt/generator.pth')
    torch.save(discriminator.state_dict(), './ckpt/discriminator.pth')
    torch.save(auxilarry_nn.state_dict(), './ckpt/auxilarry.pth')
    