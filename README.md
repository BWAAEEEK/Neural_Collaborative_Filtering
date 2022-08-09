# Neural_Collaborative_Filtering
Pytorch로 NCF 구현

## Dataset: MovieLens - 1M  
* interaction.pkl : user를 key로 가지고 해당 user가 본 영화들의 list를 value로 가지는 dictionary
* user_list.pkl: user들의 index가 저장된 list
* movie_list.pkl: 영화들의 index가 저장된 list

## Setup
* torch == 1.12.0+cu116
* tqdm == 4.62.3

## Hyper Parameter Setting
* embedding size: 100
* num_layer: 3
* hidden_size = 1024, 128, 64
* learning_rate = 0.001
* batch size: 1000



" linear layer에 activation function 적용 x "
