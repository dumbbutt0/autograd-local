# %%
words =open('names.txt','r').read().splitlines()

len(words)

min(len(w)for w in words)

max(len(w)for w in words)
# %%

b={}
for w in words:
    chs =['<S>']+ list(w) +['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram =(ch1,ch2)
        b[bigram]= b.get(bigram,0)   +1
        
#returns the tuple of key values representing the character bigrams and the counts
#sort by the count of each element
sorted(b.items(), key=lambda kv:-kv[1]) #-1 shows most likely, 1 shows furthest

# w = string emma,  w[1:] ='mma'
# %%
import torch

#creates a 3/5 array of zeroes, a.dtype == torch.float32
a = torch.zeros((3,5), dtype=torch.int32)
#%%
#sets 1 at position 1,3 (dont forget its zero indexed)
a[1,3] += 1
# %%
a
# %%
N = torch.zeros((28,28), dtype=torch.int32)
#dedupes characters
chars =sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi
stoi['<S>']=26
stoi['<E>']=27
# %%
for w in words:
    chs =['<S>']+ list(w) +['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 =stoi[ch1]
        ix2 =stoi[ch2]
        N[ix1,ix2] += 1

# %% 
itos ={i:s for s,i in stoi.items()}
itos
# %%
import matplotlib.pyplot as plt
#plt.imshow(N) == show the full array 
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')   #plot bigram text
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray') #plot number of occurences
        #.item() is needed because indexing into the arrays still returns tensor.Torch types
plt.axis('off');
# %%
N = torch.zeros((27,27), dtype=torch.int32)
#dedupes characters
chars =sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos ={i:s for s,i in stoi.items()}

for w in words:
    chs =['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 =stoi[ch1]
        ix2 =stoi[ch2]
        N[ix1,ix2] += 1

# %%
import matplotlib.pyplot as plt
#plt.imshow(N) == show the full array 
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')   #plot bigram text
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray') #plot number of occurences
        #.item() is needed because indexing into the arrays still returns tensor.Torch types
plt.axis('off');
# %%

# %%
#probability vector (flaots needed for normalization (p= p/ p.sum()))
p=N[0].float()
p= p/ p.sum()
g =torch.Generator().manual_seed(2147483647)
ix= torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()
itos[ix]
# %%
#create and see a torch Generator object
g =torch.Generator().manual_seed(2147483647)
p= torch.rand(3,generator=g) #creates random numbers using the set generator
p= p/p.sum()
#replacemt = True == when we drawn an element were allowed to draw an element and put it back into the eligible indices 
torch.multinomial(p,num_samples=1, replacement=True, generator=g)
# %%
N[0].sum().item()
# %%
g =torch.Generator().manual_seed(2147483647)

for i in range(50):
    out =[]
    ix =0
    while True:
        #trained on bigrams     bigrams == 1 number followed by another
        #   A bigram means one token followed by another token.
        p=N[ix].float()
        p= p/p.sum()
        #makes everything equally likely
        #p =torch.ones(27)/27
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
    print(''.join(out))

# %%
#P= Bigram count matrix

# a (27,1) "column" broadcasts across columns (what you want when normalizing rows);
# a bare (27,) broadcasts across rows because the missing dimension is inserted on the left.
#  keepdim=True preserves the axis you summed over so it lines up with the axis you're dividing along.

# P.sum(1)      shape (27,) == (1,27)   — 1-D vector of row sums (dim squeezed out)
#brodcasting rules go from right to left and if anything is missing fill in that direction, keedpdim enforces this
P= N.float()
#use in place operations (e.g. '/=' ) or else youll be creating new tensor objects stored in P
P /= P.sum(1, keepdim=True)  # shape (27, 1)  — column vector of row sums


# %%
g =torch.Generator().manual_seed(2147483647)
for i in range(5):
    out =[]
    ix =0
    while True:
        p= P[ix]
        #trained on bigrams
        #p=N[ix].float()
        #p= p/p.sum()
        #makes everything equally likely
        #p =torch.ones(27)/27
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
    print(''.join(out))
# %%h
#likelihood =1x1*1x2*prob
#log-likelihood = log of probability


#P= N+1000 clears all zeroes effectively eliminating inf
P= (N+1000).float()
P /= P.sum(1, keepdim=True) 
log_likelihood =0.0
n=0
for w in ["andrejq"]:
    chs =['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 =stoi[ch1]
        ix2 =stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}{prob:.4f}{logprob:.4}')
print(f'{log_likelihood=}')

nll =-log_likelihood
# normalize negative loss likelihood for average loss
print(f'{nll/n}')
# %%

#create training set of all bigrams (x,y)
xs,ys =[],[]


for w in words[:1]:
    chs =['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 =stoi[ch1]
        ix2 =stoi[ch2]
        print(ch1,ch2)
        xs.append(ix1)
        ys.append(ix2)

    xs =torch.tensor(xs)
    ys = torch.tensor(ys)
#tensor =  int64    ,   Tensor - float32

print(f'{xs}\n\n{ys}')
# %%
import torch.nn.functional as F

xenc= F.one_hot(xs, num_classes =27).float()
xenc
xenc.shape
# %%
plt.imshow(xenc)
# %%
W = torch.rand((27,27))
W
#[3,13] == 13   ,   3rd neuron at 13th collum
xenc @ W
# %%
xenc[3]
# %%
W[:,13]
# %%
(xenc[3]*W[:,13]).sum()
# %%
W = torch.rand((27,27))
(xenc @ W).exp()

#(xenc @ W).exp() makes all the negative numbers now positive  and gives us a measured count
# %%
logits =xenc @ W        #log-counts
count = logits.exp()    #equivalent N
prob - count / counts.sum(keepdims =True)
probs.shape

probs[0].sum()
# %%
