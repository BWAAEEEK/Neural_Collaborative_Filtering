import pickle
from tqdm import tqdm
import pandas as pd
import pprint

movie = pd.read_table('movies.dat', sep='::', names=['movie_id', 'movie_name', 'tag'], engine='python')
rating = pd.read_table("ratings.dat", sep="::",
                           names=["user_id", "movie_id", "rating", "timestamp"], engine='python')

print(movie)
print(rating)

user_list = list(set(rating["user_id"]))
movie_list = list(set(rating["movie_id"]))

print(max(user_list))
print(max(movie_list))


print(len(user_list))
print(len(movie_list))
# with open("../../user_list.pkl", "wb") as f:
#     pickle.dump(user_list, f)
#
# with open("../../movie_list.pkl", "wb") as f:
#     pickle.dump(movie_list, f)


# interaction = {user: set() for user in user_list}

# for seq in tqdm(rating.iloc(), total=1000208):
#     interaction[seq[0]].add(seq[1])

# interaction = {user: list(movies) for user, movies in interaction.items()}


# with open("./interaction.pkl", "wb") as f:
#     pickle.dump(interaction, f)
