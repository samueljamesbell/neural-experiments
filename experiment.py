from feed_forward_network import FeedForwardNet
import iris_data_loader
import utils


_TRAINING_EPOCHS = 1
_LEARNING_RATE = 0.0001

_LAYERS = [4, 3, 3]
# Input layer size = 4
# Single hidden layer size = 3
# Output layer size = 3


def _main():
    X_train, Y_train = iris_data_loader.training_data()
    X_test, Y_test = iris_data_loader.test_data()
    n = FeedForwardNet(_LAYERS)

    Y_train_categorical = utils.to_categorical_identity(
            Y_train, iris_data_loader.LABELS)
    Y_train_prediction = n.train(
            X_train, Y_train_categorical,
            training_epochs=_TRAINING_EPOCHS,
            learning_rate=_LEARNING_RATE)
    train_accuracy = utils.accuracy(Y_train_prediction, Y_train_categorical)
    print('Training accuracy: {}'.format(train_accuracy))

    Y_test_categorical = utils.to_categorical_identity(
            Y_test, iris_data_loader.LABELS)
    Y_test_prediction = n.test(X_test)
    test_accuracy = utils.accuracy(Y_test_prediction, Y_test_categorical)
    print('Test accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    _main()
