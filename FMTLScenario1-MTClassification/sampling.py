"This file contains different functions for sampling data"
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

"""Start"""
'''
This function splits the dataset evenly and randomly among the users,
creating an IID (Independent and Identically Distributed) data scenario.

:param dataset: specifies the dataset to be split among users
:param num_users: specifies the number of users among which the dataset is to be split
:return: A dictionary where the keys as user IDs and values are sets of image indices assigned to each user

'''
def mnist_iid(dataset, num_users):

 num_sets = int(len(dataset)/num_users)

 # Define empty dictionary
 dict_users = {}

 # List of all indices in the dataset
 all_idxs = [i for i in range(len(dataset))]

 # For loop to assign image indices to each user
 for i in range(num_users):
  # np.random.choice selects 'num_sets' random indices from 'all_idxs' (without replacement)
  # These indices are assigned to user i in the dictionary and that is converted to a set
  dict_users[i] = set(np.random.choice(all_idxs,num_sets,replace=False))

  # Remove the assigned indices from 'all_idxs' to avoid the reuse in next iteration
  # all_idxs is converted to a set to do the subtraction and then the resultant is converted back to list
  all_idxs = list(set(all_idxs) - dict_users[i])

 return dict_users


# Testing the above function
# mnist_train = datasets.MNIST('./data', train= True, download=True, transform=transforms.ToTensor())
# users = mnist_iid(mnist_train,10)
#
# print(users)


"""End"""

"""Start"""

'''
This function splits the dataset into shards and shards are assigned to users
such that data with each user is skewed. Each user gets the same number of shards and hence the images.
For example - one user gets data mostly 0s and 1s
and the other gets mostly 2s and 3s, and so on.
:param dataset: specifies the dataset to be split among users
:param num_users: specifies the number of users among which the dataset is to be split
:return: A dictionary where the keys as user IDs and values are sets of image indices assigned to each user
'''

def mnist_non_iid(dataset,num_users):

 # 60k training images in MNIST --> 200 images/shard X 300 shards
 num_shards, num_imgs = 300, 200

 # Generate a numpy array containing 0 to 59999
 idxs = np.arange(num_shards*num_imgs)

 # Create a list containing indices of shards
 idx_shard = [i for i in range(num_shards)]

 # Initialize a dictionary with user index as keys and empty arrays as values
 dict_users = {i: np.array([], dtype = np.int64) for i in range (num_users)}

 # Get labels
 labels = dataset.targets.numpy()

 # Sort labels
 # Vertically stacking the numpy arrays idxs and labels to create a 2-row matrix
 idxs_labels = np.vstack((idxs, labels))

 # Sorts the entries (image indices) by labels
 idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]

 # Extract the sorted indices
 idxs = idxs_labels[0,:]

 # Randomly select two shards from the list of the available shards, for each user
 for i in range(num_users):
  rand_set = set(np.random.choice(idx_shard, 2, replace=False))

  # Remove the already picked shard from the idx_shard
  idx_shard = list(set(idx_shard) - rand_set)

  # Assign image indices from all the shards to a user i
  for rand in rand_set:

   # Calculate the starting and the ending index of the shard in the full idxs array
   start = rand * num_imgs
   end = (rand+1) * num_imgs

   # Extract the image indices for this shard
   shard_idxs = idxs[start:end] # This will be a 1D numpy array of 200 elements (if num_imgs = 200)

   # Get the current indices already assigned to the user
   current_user_data = dict_users[i]

   # Concatenate then new shard indices to the user's current data
   updated_user_data = np.concatenate((current_user_data,shard_idxs), axis = 0)

   # Save back to the dictionary
   dict_users[i] = updated_user_data

   # dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis = 0)

 return dict_users


"""End"""

# Testing the above function
# mnist_train = datasets.MNIST('./data', train= True, download=True, transform=transforms.ToTensor())
# users = mnist_non_iid(mnist_train,100)
# #
# print(users)


"""Start"""

'''
This function splits the dataset into shards and shards are assigned to users
such that data with each user is skewed. Each user gets different number of shards and hence the images.
For example - one user gets data mostly 0s and 1s
and the other gets mostly 2s and 3s, and so on.
:param dataset: specifies the dataset to be split among users
:param num_users: specifies the number of users among which the dataset is to be split
:return: A dictionary where the keys as user IDs and values are (unequal) sets of image indices assigned to each user
'''

def mnist_noniid_unequal(dataset, num_users):

 # 60k training images in MNIST --> 50 images/shard X 1200 shards
 num_shards, num_imgs = 1200, 50

 # Generate a numpy array containing 0 to 59999
 idxs = np.arange(num_shards * num_imgs)

 # Create a list containing indices of shards
 idx_shard = [i for i in range(num_shards)]

 # Initialize a dictionary with user index as keys and empty arrays as values
 dict_users = {i: np.array([], dtype = np.int64) for i in range(num_users)}

 # Get labels
 labels = dataset.targets.numpy()

 # Sort labels
 # Vertically stacking the numpy arrays idxs and labels to create a 2-row matrix
 idxs_labels = np.vstack((idxs, labels))

 # Sorts the entries (image indices) by labels
 idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

 # Extract the sorted indices
 idxs = idxs_labels[0, :]

 # Define minimum and maximum shards that can be assigned to a user
 min_shard = 1
 max_shard = 30

 # Dividing the shards between the users, at random. Each client gets different (random) number of shards

 # Randomly generate shard sizes for each user within a given range
 random_shard_size = np.random.randint(min_shard, max_shard+1, size=num_users)

 # Normalize so the total matches the num_shards
 random_shard_size = np.around(random_shard_size/sum(random_shard_size) * num_shards)

 # Convert to integers
 random_shard_size = random_shard_size.astype(int)

 # Correction to ensure that the sum remains equal to num_shards despite the round-off
 diff = num_shards - np.sum(random_shard_size)

 # Adjust the first diff element - +1 if under, -1 if over
 for i in range(abs(diff)):
  index = i % num_users
  random_shard_size[index] += np.sign(diff)


  # Assign the shards randomly to the users
  if sum(random_shard_size) > num_shards:

   for i in range(num_users):

    # First assign 1 shard to each user to ensure each user has one shard
    rand_set = set(np.random.choice(idx_shard,1,replace=False))
    idx_shard = list(set(idx_shard) - rand_set)
    for rand in rand_set:
     dict_users[i]=np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand+1)*num_imgs]), axis = 0)

   random_shard_size = random_shard_size - 1

   # Randomly assign the remaining shards

   for i in range(num_users):
    if len(idx_shard) == 0:
     continue
    shard_size = random_shard_size[i]

    if shard_size > len(idx_shard):
     shard_size = len(idx_shard)

    rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))

    idx_shard = list(set(idx_shard) - rand_set)

    for rand in rand_set:
     dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand +1) * num_imgs]), axis =0)

  else:

   for i in range(num_users):
    shard_size = random_shard_size[i]
    rand_set = set(np.random.choice(idx_shard, shard_size, replace= False))
    idx_shard = list(set(idx_shard) - rand_set)

    for rand in rand_set:
     dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand +1) * num_imgs]), axis=0)

   if len(idx_shard) > 0:

    # Add leftover shards to the user with minimum images
    shard_size = len(idx_shard)

    # Add the remaining shards to the user with the lowest data
    k= min(dict_users, key=lambda x: len(dict_users.get(x)))
    rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
    idx_shard = list(set(idx_shard) - rand_set)

    for rand in rand_set:
     dict_users[k] = np.concatenate((dict_users[k], idxs[rand *num_imgs : (rand+1)*num_imgs]), axis=0)

 return dict_users

""""End"""


"""Start"""

# Second type of skewed dataset
def custom_skewed_partition(dataset, num_users, seed=None):
 """
 Custom non-IID and equal partition for clients:
  - Client 0: 20% of digit 2, 80% from others (excluding 2)
  - Client 1: 15% each of digits 1, 2, 3, and 55% from others (excluding 1,2,3)
  - Client 2: 20% of digit 6, 80% from others (excluding 6)
 Ensures: no overlap across clients; exact samples_per_client per client.
 """
 import numpy as np
 rng = np.random.default_rng(seed)
 targets = dataset.targets.numpy()
 total_samples = len(targets)
 samples_per_client = total_samples // num_users  # 60000 // 3 = 20000

 # Pool of available indices per digit (mutable)
 available_by_digit = {d: np.where(targets == d)[0].tolist() for d in range(10)}
 for d in available_by_digit:
  rng.shuffle(available_by_digit[d])

 # Bias config per client
 bias_config = {
  0: {"include": {2: 0.11}, "exclude": [2]},
  1: {"include": {1: 0.15, 2: 0.12, 3: 0.15}, "exclude": [1, 2, 3]},
  2: {"include": {6: 0.20}, "exclude": [6]},
 }

 def take_from_digit(digit, k):
  """Pop k indices from this digit’s pool."""
  k = min(k, len(available_by_digit[digit]))
  taken = available_by_digit[digit][:k]
  del available_by_digit[digit][:k]
  return taken

 dict_users = {}

 for client_id in range(num_users):
  config = bias_config[client_id]
  include = config["include"]
  exclude = set(config["exclude"])

  client_indices = []

  # 1) Add biased samples
  for digit, ratio in include.items():
   need = int(round(ratio * samples_per_client))
   client_indices.extend(take_from_digit(digit, need))

  # 2) Fill the remainder from "others" digits (excluding `exclude`)
  remaining_needed = samples_per_client - len(client_indices)
  if remaining_needed > 0:
   # Build a unified pool of (digit, count_available) for allowed digits
   allowed_digits = [d for d in range(10) if d not in exclude]
   # Keep drawing until we hit the target or pools are empty
   while remaining_needed > 0 and any(len(available_by_digit[d]) > 0 for d in allowed_digits):
    # Pick a digit with available samples
    d_choices = [d for d in allowed_digits if len(available_by_digit[d]) > 0]
    d = rng.choice(d_choices)
    take = min(remaining_needed, len(available_by_digit[d]), 512)  # draw in chunks for speed
    client_indices.extend(take_from_digit(d, take))
    remaining_needed -= take

  # Final safety check: if we couldn’t fill (ran out of data), trim to exact length
  client_indices = client_indices[:samples_per_client]
  dict_users[client_id] = np.array(client_indices, dtype=np.int64)

 # Optional sanity checks
 # - no overlap across clients
 all_ids = np.concatenate([dict_users[i] for i in range(num_users)])
 assert len(all_ids) == len(set(all_ids.tolist())), "Overlap found between client splits!"
 # - exactly equal sizes
 for i in range(num_users):
  assert len(dict_users[i]) == samples_per_client, f"Client {i} size mismatch."

 return dict_users


from collections import Counter
import numpy as np

def print_label_distribution(dict_users, dataset):
    for client_id, indices in dict_users.items():
        labels = dataset.targets[indices]
        label_counts = Counter(labels.numpy())
        total_samples = len(indices)
        print(f"\nClient {client_id} label distribution (Total:{total_samples} samples):")
        for label in sorted(label_counts):
            print(f"  Label {label}: {label_counts[label]} samples")


# mnist_train = datasets.MNIST('./data', train= True, download=True, transform=transforms.ToTensor())
# dict_users = custom_skewed_partition(mnist_train, num_users=3)
# print_label_distribution(dict_users, mnist_train)