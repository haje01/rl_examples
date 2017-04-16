# Billiard


## 2ball.py

### Generate play data

Generate play data of 1000 shots for fully connected net.

    $ python 2ball.py gendata --shotcnt 1000 data/2ball_fc --encout

Generate play data of 1000 shots for CNN.

    $ python 2ball.py gendata --shotcnt 1000 data/2ball_cnn

### Train neural net from play data

Train fully connected net from play data with 1000 episode, and save it.

    $ python 2ball.py train fc data/2ball_fc --episode 1000 --modelpath models/2ball_fc

Train CNN from play data with 1000 episode, and save it.

    $ python 2ball.py train cnn data/2ball_fc --episode 1000 --modelpath models/2ball-cnn


### Bot play by trained model data.

Bot play by FC model.

    $ python 2ball.py botplay models/2ball_fc

Bot play by CNN model.

    $ python 2ball.py botplay models/2ball_cnn
