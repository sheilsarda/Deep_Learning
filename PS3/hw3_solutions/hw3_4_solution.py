#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.optim
import torch.nn


# Download data
book = "./book-war-and-peace.txt"
if not os.path.exists(book):
    os.system("wget https://raw.githubusercontent.com/mmcky/"
              "nyu-econ-370/master/notebooks/data/book-war-and-peace.txt")

with open(book, "r") as fp:
    content = fp.readlines()

content = ''.join(content)
content = content.replace('\n', '')

chars = list(set(content))
chars = sorted(chars)
char_num = len(chars)
print("List of characters and counts")
print(len(chars))
print(chars)


def get_counts(content, chars):
    char_count = {}
    for ch in chars:
        char_count[ch] = 0
    for ch in content:
        char_count[ch] += 1
    print(char_count)


print("Counts of individual characters")
get_counts(content, chars)

# Question 4(a)
###############
one_hot_map = {}
for idx, ch in enumerate(chars):
    one_hot_map[ch] = idx


def get_vector(text):
    indices = [one_hot_map[ch] for ch in text]
    vectorized = np.eye(char_num)[indices]
    return vectorized, indices


# Convert text to vectors
vectorized_text, index_text = get_vector(content)
vectorized_text = torch.Tensor(vectorized_text)
index_text = torch.Tensor(index_text).long()

print(len(vectorized_text))
NUM_TRAIN = 3000000
NUM_VAL = 5000 + NUM_TRAIN
train_x = vectorized_text[:NUM_TRAIN]
train_y = index_text[:NUM_TRAIN]

val_x = vectorized_text[NUM_TRAIN:NUM_VAL]
val_y = index_text[NUM_TRAIN:NUM_VAL]


def get_batch(counter, gpu, text, labels):
    x_batch = text[counter * 25: (counter+1) * 25]
    y_batch = labels[counter * 25 + 1: (counter+1) * 25 + 1]
    x_batch = x_batch.reshape(-1, char_num)
    ln = min(len(x_batch), len(y_batch))
    x_batch, y_batch = x_batch[:ln], y_batch[:ln]
    if gpu:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    return (x_batch, y_batch)


# Question 4(b)
###############
x, y = get_batch(1, False, vectorized_text, index_text)
print("Xshape", x.shape)
print("Target", y)

# Training params
num_batches = len(train_y) // 25
num_val_batches = len(val_y) // 25
use_gpu = torch.cuda.is_available()
learning_rate = 4e-4
EPOCHS = 15
HID_SIZE = 200
cosine = True


class charRNN(torch.nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(charRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=char_num,
                                hidden_size=hid_dim,
                                num_layers=1,
                                nonlinearity='tanh',
                                bias=True)
        # add another embedding layer to improve training (self.embed is an
        # extra layer)
        self.embed = torch.nn.Linear(char_num, char_num)
        self.linear = torch.nn.Linear(hid_dim, out_dim)
        self.hid_dim = hid_dim

    def forward(self, x, hidden):
        embed = self.embed(x)
        embed = embed.view(-1, 1, char_num)
        out, hidden = self.rnn(embed, hidden)
        out = out.view(-1, self.hid_dim)
        out = self.linear(out)
        return out, hidden


def unroll(steps=25, start=0, sample=False):
    hidden = torch.zeros(1, 1, HID_SIZE)
    net.eval()
    # Start with random character
    data = train_x[start].reshape(1, 1, -1)
    pred = '' + chars[train_y[start]]
    for st in range(steps):
        hidden = hidden.cpu().detach()
        if use_gpu:
            hidden = hidden.cuda()
            data = data.cuda()
        output, hidden = net(data, hidden)

        if sample:
            output = torch.nn.functional.softmax(output, dim=1)
            output = np.reshape(output.cpu().detach().numpy(), (-1))
            out_char = np.random.choice(list(range(char_num)), p=output)
        else:
            out_char = torch.argmax(output)

        pred = pred + chars[out_char]
        data = torch.zeros(1, 1, char_num)
        data[0, 0, out_char] = 1
    print("Unrolling RNN: %s" % pred)


net = charRNN(HID_SIZE, char_num)
if use_gpu:
    net = net.cuda()
    print("Using GPU")
else:
    print("Using CPU")
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optim, milestones=[10, 20, 25], gamma=0.2)

for ep in range(EPOCHS):
    print("Epoch")
    hidden = torch.zeros(1, 1, HID_SIZE)
    train_loss = 0.0
    for it in range(num_batches):
        net.train()
        hidden = hidden.cpu().detach()
        if use_gpu:
            hidden = hidden.cuda()
        optim.zero_grad()
        # Train model and make updates
        data, target = get_batch(it, use_gpu, train_x, train_y)
        output, hidden = net(data, hidden)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optim.step()
        train_loss += loss.item()

        # Accumulate training statistics
        it = num_batches*ep + it
        if it % 1000 == 0:
            train_loss = train_loss / 1000
            print("Loss at %d: %f" % (it, train_loss))
            train_loss = 0.0
            unroll(100, 0)

        # Every 1000 steps, evaluate model
        if it % 1000 == 0:
            net.eval()
            hidden_val = torch.zeros(1, 1, HID_SIZE)
            total_loss = 0.0
            total_acc = 0.0
            for t_it in range(num_val_batches):
                hidden_val = hidden_val.cpu().detach()
                if use_gpu:
                    hidden_val = hidden_val.cuda()
                val_data, val_target = get_batch(t_it, use_gpu, val_x, val_y)
                output, hidden_val = net(val_data, hidden_val)
                loss = criterion(output, val_target)
                total_loss += loss.item()
                out_val = np.reshape(output.cpu().detach().numpy(),
                                     (-1, char_num))
                total_acc = np.mean(np.argmax(out_val, axis=1) ==
                                    val_target.cpu().numpy())
            total_loss = total_loss / num_val_batches
            total_acc = total_acc / num_val_batches * 100
            print("Validation loss at %d: %f %f" % (it, total_loss, total_acc))

    if cosine:
        lr_scheduler.step()


unroll(500, 0, False)
print("----")
unroll(500, 0, True)
print("----")
unroll(500, 0, True)
print("----")
unroll(500, 0, True)
torch.save(net, "net.pth")
