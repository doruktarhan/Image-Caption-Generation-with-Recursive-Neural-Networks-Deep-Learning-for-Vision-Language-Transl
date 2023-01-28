import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torch.nn.functional import one_hot

import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#Definitions 

def save_images(url_list, save_path):
    success_count = 0
    unsuccess_count = 0
    #f = open('unsuccess_train_list.txt', 'w') #uncomment for train images
    f = open('unsuccess_test_list.txt', 'w') #uncomment for test images
    for i in range(url_list.shape[0]):
        path = save_path + str(i) + '.jpg'
        imgURL = url_list[i].decode('UTF-8')
        try:
            urllib.request.urlretrieve(imgURL, path)
            success_count += 1
        except:
            unsuccess_count += 1
            f.write(str(i) + '\n')
    print(success_count, ' images downloaded successfully.')
    print(unsuccess_count, ' images are failed to download.')
    f.close()


def broke_url_delete(filename, imid_h5, cap_h5, path):
    '''
        inputs:
            -filename: unsuccess_list.txt bla bla
            -imid_h5: train/test_imid_h5 dataset
            -cap_h5: train/test_cap_h5 dataset
            -path: path for saving the tensors as .tar file
    '''
    imid = np.zeros(imid_h5.shape)
    cap = np.zeros(cap_h5.shape, dtype = int)
    for i in range(imid_h5.shape[0]):
        imid[i] = imid_h5[i] - 1
        cap[i] = cap_h5[i]
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        ind_list = np.where(imid == int(line))
        imid = np.delete(imid, ind_list, 0)
        cap = np.delete(cap, ind_list, 0)
    f.close()
    imid_tensor = torch.from_numpy(imid)
    cap_tensor = torch.from_numpy(cap)
    data = (imid_tensor, cap_tensor)
    torch.save(data, path)


def train_val_split(dset, perm_ind, percentage = 0.85):
    split_ind = int(dset.shape[0] * percentage)
    tr_indices = perm_ind[0:split_ind]
    val_indices = perm_ind[split_ind: dset.shape[0]]
    tr_dset = dset[tr_indices]
    val_dset = dset[val_indices]
    return tr_dset, val_dset


def mere_lstm_train2(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0
    for batch_num, data in enumerate(dataloader):
        latents, caps = data #latents aslında features gibi de düşünülebilir  latents : (N, latent_dim)  caps : (N, seq_len)
        optimizer.zero_grad()
        outputs = model(latents, caps) # outputs : (seq_len, N, voc_size)
        loss = criterion(outputs.reshape(-1,outputs[1:].shape[-1]), torch.swapaxes(caps, 0, 1).reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def mere_lstm_evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_num, data in enumerate(dataloader):
            latents, caps = data #latents aslında features gibi de düşünülebilir  latents : (N, latent_dim)  caps : (N, seq_len)
            outputs = model(latents, caps, train_bool = False) # outputs : (seq_len, N, voc_size)
            loss = criterion(outputs.reshape(-1,outputs[1:].shape[-1]), torch.swapaxes(caps, 0, 1).reshape(-1))
            running_loss += loss.item()
    return running_loss / len(dataloader)



    
    



def mere_lstm_train(model, dataloader, learning_rate, num_epoch, save_filename, device):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    for _, param in model.named_parameters():
        param.requires_grad = True
    model.train()
    for epoch in range(num_epoch):
        for batch_num, data in enumerate(dataloader):
            # print('inside train func.\nbatch_num:', batch_num)
            # print('latent shape:', data[0].shape)
            # print('caps shape:', data[1].shape)
            latents, caps = data #latents aslında features gibi de düşünülebilir  latents : (N, latent_dim)  caps : (N, voc_size
            optimizer.zero_grad()
            outputs = model(latents, caps) # outputs : (seq_len, N, voc_size)
            # print('\nback in train\noutputs shape:', outputs.shape)
            # print('caps shape:', caps.shape)
            # print('one_hot(caps) shape:', one_hot(caps, num_classes = 1004).shape)
            # print('outputs.dtype:', outputs.dtype)
            # print('one_hot(caps).dtype:', one_hot(caps, num_classes = 1004).dtype)

            #loss = criterion(outputs.float(), one_hot(caps, num_classes = 1004).float())
            loss = criterion(outputs.reshape(-1,outputs[1:].shape[-1]), torch.swapaxes(caps, 0, 1).reshape(-1))
            
            loss.backward()
            optimizer.step()


            if batch_num % 500 == 0:
                print('Epoch : ', epoch, ',', batch_num, "'th Batch ---> Loss = ", loss.item())
    print('\n Training Finished \n')
    checkpoint = {
        'epoch' : num_epoch,
        'model_state' : model.state_dict(),
        'optim_state' : optimizer.state_dict()
    }
    #torch.save(checkpoint, save_filename)


# classes 

class EncoderCNN_Resnet(nn.Module):
    def __init__(self,embed_size,train_CNN = False):
        super(EncoderCNN_Resnet,self).__init__()
        self.train_CNN = train_CNN
        self.CNNmodel =  torchvision.models.resnet18(pretrained=True)
        self.CNNmodel.fc = nn.Linear(self.CNNmodel.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        
        
    def forward(self,images):
        features = self.CNNmodel(images)
        
        for name , param in self.CNNmodel.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else: 
                param.requires_grad = self.train_CNN
                
        features = self.relu(features)       
        #return self.dropout(self.relu(features))
        return features
    
   
class EncoderCNN_v3inception(nn.Module):
    def __init__(self,embed_size,train_CNN = False):
        super(EncoderCNN_v3inception,self).__init__()
        self.train_CNN = train_CNN
        self.CNNmodel =  torchvision.models.inception_v3(pretrained=True)
        self.CNNmodel.fc = nn.Linear(self.CNNmodel.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        
        
    def forward(self,images):
        features = self.CNNmodel(images)
        
        for name , param in self.CNNmodel.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else: 
                param.requires_grad = self.train_CNN
         
        features = self.relu(features)
        #return self.dropout(self.relu(features))
        return features   
    
class EncoderCNN_VGGNet(nn.Module):
    def __init__(self,embed_size,train_CNN = False):
        super(EncoderCNN_v3inception,self).__init__()
        self.train_CNN = train_CNN
        self.CNNmodel =  torchvision.models.vgg19_bn(pretrained=True)
        self.CNNmodel.fc = nn.Linear(self.CNNmodel.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        
        
    def forward(self,images):
        features = self.CNNmodel(images)
        
        for name , param in self.CNNmodel.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else: 
                param.requires_grad = self.train_CNN
        features = self.relu(features)        
        #return self.dropout(self.relu(features))
        return features  

class caption_generator():

    def __init__(self, file_name, key_name):
        words_df = pd.read_hdf(file_name, key_name)
        words_dict = words_df.to_dict('split')
        print(words_dict.keys())
        self.words = words_dict['columns']
        self.word_numbers = words_dict['data']

    def form_sentence(self, cap):
        for i in range(cap.shape[0]):
            
            ind = self.word_numbers[0].index(int(cap[i]))
            print(self.words[ind], end = ' ')
            # if self.words[ind] == 'x_END_':
            #     print()
            #     break
            # if self.words[ind] != 'x_START_':
            #     print(self.words[ind], end = ' ')



class MereLSTMDataset(Dataset):
    def __init__(self, im_id, caps, latent_features, transform = None, target_transform = None):
        super().__init__()
        self.im_id = im_id
        self.caps = caps
        self.lat_feat = latent_features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.im_id.shape[0]
    
    def __getitem__(self, index):
        cap = self.caps[index]
        lat_ind = int(self.im_id[index])
        latent = self.lat_feat[lat_ind]
        if self.transform:
            latent = self.transform(latent)
        if self.target_transform:
            cap = self.target_transform(cap)
        return latent, cap

        

        


class DecoderLSTM(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, num_of_lay, device):
        super(DecoderLSTM, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_lay = num_of_lay
        self.embed = nn.Embedding(voc_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, num_of_lay)
        self.linear = nn.Linear(hid_size, voc_size)

    def forward(self, latent, caps, train_bool=True):

        outputs = torch.zeros(caps.shape[-1], caps.shape[0], self.voc_size).to(self.device) #outputs : (seq_len, batch_size(N), voc_size)
        hidden = latent.unsqueeze(0) #(hid_size) the latent should be with shape (hid_size)
        #cell = latent
        #print('\ninside decoder lstm hidden shape:', hidden.shape)
        cell = torch.zeros(hidden.shape[1], self.hid_size).unsqueeze(0).to(self.device)
        #print('inside decoder lstm hidden shape[1]:', hidden.shape[1])
        inp = caps[:,0] # inp : (batch_size(N))
        for i in range(1, caps.shape[1]):
            inp = inp.unsqueeze(0) # inp : (1, batch_size(N))
            emb = self.embed(inp) # emb : (1, N, emb_size)
            #print('\ninside decoderlstm forward for loop\nhidden shape:', hidden.shape,'\ncell shape:',cell.shape,'\n')
            out, (hidden, cell) = self.lstm(emb, (hidden, cell)) # out : (1, N, hid_size)   hid, cell : (num_lay, N, hid_size)
            output = self.linear(out.squeeze(0)) #output : (N, voc_size)
            outputs[i] = output
            max_val = output.argmax(1) # max_val : (N)
            inp = caps[:,i] if train_bool else max_val
        return outputs

        # print('\ninside decoderlstm\nlatent shape:',latent.shape)
        # print('caps shape:',caps.shape)
        # print('caps dtype:',caps.dtype)
        # print('embed(caps) shape:',self.embed(caps).shape, '\n')
        # print('embed(caps)[1:] shape:',self.embed(caps)[:,1:,:].shape, '\n')

        # embeds = torch.cat((latent.unsqueeze(1), self.embed(caps)[:,1:,:]), dim=1)

        # print('embeds shape:', embeds.shape)
        # hids,_ = self.lstm(embeds)
        # outs = self.linear(hids)
        # return outs


class BareDecoder(nn.Module):
    def __init__(self, feat_size, voc_size, emb_size, hid_size, num_of_lay, device):
        super(BareDecoder, self).__init__()
        self.device = device
        self.linear = nn.Linear(feat_size, emb_size)
        self.decoderLSTM = DecoderLSTM(voc_size, emb_size, hid_size, num_of_lay,self.device)

    def forward(self, features, caps, train_bool = True):
        #print('\ninside baredecoder\nfeatures shape:', features.shape)
        #print('caps shape:', caps.shape)

        latent = self.linear(features)
        #print('latent shape:', latent.shape)
        outs = self.decoderLSTM(latent, caps, train_bool)        
        #print('outs shape:', outs.shape)
        return outs






file1 = h5py.File('eee443_project_dataset_test.h5', 'r')
file2 = h5py.File('eee443_project_dataset_train.h5', 'r')
print('file1 keys :', file1.keys())
print('file2 keys :', file2.keys())

train_cap_h5 = file2.get('train_cap')
train_imid_h5 = file2.get('train_imid')
train_ims_h5 = file2.get('train_ims')
train_url_h5 = file2.get('train_url')
word_code_h5 = file2.get('word_code')
test_caps_h5 = file1.get('test_caps')
test_imid_h5 = file1.get('test_imid')
test_ims_h5 = file1.get('test_ims')
test_url_h5 = file1.get('test_url')




'''
#uncomment this codeblock for downloading the images if you have not done it already
images_path = '/home/aniyazi/Downloads/train_ims/' #uncomment for downloading train images
images_path = '/home/aniyazi/Downloads/test_ims/' #uncomment for downloading test images
save_images(train_url_h5, images_path) #uncomment for downloading train images
save_images(test_url_h5, images_path) #uncomment for downloading test images
'''



tr_path = 'train_data_tensor.pth'
test_path = 'test_data_tensor.pth'
'''
#uncomment this codeblock if you have not deleted the samples with broken URL
tr_filename = 'unsuccess_train_list.txt'
test_filename = 'unsuccess_test_list.txt'
broke_url_delete(tr_filename, train_imid_h5, train_cap_h5, tr_path)
broke_url_delete(test_filename, test_imid_h5, test_caps_h5, test_path)
'''
(train_imid, train_cap) = torch.load(tr_path)
(test_imid, test_cap) = torch.load(test_path)

print('\nShapes of the training and test samples before deleting the samples with broken URL')
print('Shape of train_imid_h5: ', train_imid_h5.shape)
print('Shape of train_cap_h5: ', train_cap_h5.shape)
print('Shape of test_imid_h5: ', test_imid_h5.shape)
print('Shape of test_caps_h5: ', test_caps_h5.shape)
print('\nShapes of the training and test samples after deleting the samples with broken URL')
print('Shape of train_imid of type torch.tensor: ', train_imid.shape)
print('Shape of train_cap of type torch.tensor: ', train_cap.shape)
print('Shape of test_imid of type torch.tensor: ', test_imid.shape)
print('Shape of test_cap of type torch.tensor: ', test_cap.shape,'\n')


#splitting training and validation sets
np.random.seed(42)
perm_ind = np.random.permutation(train_imid.shape[0])

train_imid, val_imid = train_val_split(train_imid, perm_ind)
train_cap, val_cap = train_val_split(train_cap, perm_ind)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#forming dataset and dataloader objects for mere lstm models
train_im_dir = '/home/aniyazi/Downloads/train_ims'
mere_lstm_train_set = MereLSTMDataset(train_imid, train_cap, train_ims_h5)
mere_lstm_val_set = MereLSTMDataset(val_imid, val_cap, train_ims_h5)

mere_lstm_train_dataloader = DataLoader(dataset = mere_lstm_train_set, batch_size = 128, shuffle = True, num_workers = 2)
mere_lstm_val_dataloader = DataLoader(dataset = mere_lstm_val_set, batch_size = 128, shuffle = True, num_workers = 2)


#train mere LSTM
"""
learning_rate = 0.01

model = BareDecoder(feat_size=512, voc_size=1004, emb_size=256, hid_size=256, num_of_lay=1, device=device)

model = model.to(device)
for _, param in model.named_parameters():
    param.requires_grad = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

lowest_val_loss = float('inf')
epoch_num = 40
save_filename = 'first_model_working_doruk_version.pth'

for epoch in range(epoch_num):

    tr_loss = mere_lstm_train2(model, mere_lstm_train_dataloader, optimizer, criterion)
    val_loss = mere_lstm_evaluate(model, mere_lstm_val_dataloader, criterion)
    if tr_loss < lowest_val_loss:
        lowest_val_loss = tr_loss
        
        checkpoint = {
        'epoch' : epoch,
        'model_state' : model.state_dict(),
        'optim_state' : optimizer.state_dict()
        }}
        
        torch.save(model,save_filename)

        #torch.save(checkpoint, save_filename)
    print('Epoch: ', epoch, ' --->  Training Loss = ', tr_loss, ' , Validation Loss = ', val_loss)

"""
"""
saved_filename = 'first_model_working.pth'
loaded_checkpoint = torch.load(saved_filename)
model.load_state_dict(checkpoint)
"""
# save_filename = 'first_model.pth'
# mere_lstm_train(model, mere_lstm_train_dataloader, 0.00001, 60, save_filename=save_filename, device=device)
save_filename_working = 'working_model.pth'

" Test için deniyotuz"
"""
(train_imid, train_cap) 
(test_imid, test_cap)
"""
"""
sample_ind = 2


caps = test_cap[sample_ind]
lat_ind = int(test_imid[sample_ind])
#caps = test_cap[sample_ind]
latent = test_ims_h5[lat_ind]
model = BareDecoder(feat_size=512, voc_size=1004, emb_size=256, hid_size=256, num_of_lay=1, device=device)
#loaded_checkpoint = torch.load('first_model_new.pth')
#model.load_state_dict(loaded_checkpoint["model_state"])
model=torch.load(save_filename_working)
model.eval()
#torch.save(model,save_filename_working)
outputs = model(torch.from_numpy(latent).unsqueeze(0), caps.unsqueeze(0), train_bool = False).squeeze(1) # outputs : (seq_len, N, voc_size)
print('outputs shape:', outputs.shape)
outputs = outputs.argmax(1)

#creating caption_generator class for using it for generating captions by using the vocab dict.
filename = 'eee443_project_dataset_train.h5'
filename2 = 'eee443_project_dataset_test.h5'
cap_gen = caption_generator(filename, 'word_code')

#yrk = sample_ind
#ind = int(test_imid[yrk])
# print('Displayed image is '+ str(ind) + '.jpg')
# print('True Captions of Image is: \n')
#cap = test_cap[yrk]

# print('cap shape:', cap.shape)
img = plt.imread('/home/aniyazi/Downloads/test_ims/' + str(lat_ind) + '.jpg')
#print('True caption : ')
#cap_gen.form_sentence(caps)
print('\nEstimated Caption : ')
cap_gen.form_sentence(outputs)
plt.imshow(img)
plt.show()
"""

"""     Train için yapilan kisim """

sample_ind = 279

cap = train_cap[sample_ind]
lat_ind = int(train_imid[sample_ind])
latent = train_ims_h5[lat_ind]
model = BareDecoder(feat_size=512, voc_size=1004, emb_size=256, hid_size=256, num_of_lay=1, device=device)
#loaded_checkpoint = torch.load('first_model_new.pth')
#model.load_state_dict(loaded_checkpoint["model_state"])
model=torch.load(save_filename_working)
model.eval()
outputs = model(torch.from_numpy(latent).unsqueeze(0), cap.unsqueeze(0), train_bool = False).squeeze(1) # outputs : (seq_len, N, voc_size)
print('outputs shape:', outputs.shape)
outputs = outputs.argmax(1)

#creating caption_generator class for using it for generating captions by using the vocab dict.
filename = 'eee443_project_dataset_train.h5'
cap_gen = caption_generator(filename, 'word_code')

yrk = sample_ind
ind = int(train_imid[yrk])
# print('Displayed image is '+ str(ind) + '.jpg')
# print('True Captions of Image is: \n')
cap = train_cap[yrk]
# print('cap shape:', cap.shape)
img = plt.imread('/home/aniyazi/Downloads/train_ims/' + str(ind) + '.jpg')
print('True caption : ')
cap_gen.form_sentence(cap)
print('\nEstimated Caption : ')
cap_gen.form_sentence(outputs)
plt.imshow(img)
plt.show()
