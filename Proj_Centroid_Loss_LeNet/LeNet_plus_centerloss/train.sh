chmod u+x MNIST_train.py
RESTORE=false LEARNING_RATE=0.3 LAMBDA=0.0 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
RESTORE=true LEARNING_RATE=0.1 LAMBDA=0.1 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
RESTORE=true LEARNING_RATE=0.1 LAMBDA=1.0 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
RESTORE=true LEARNING_RATE=0.1 LAMBDA=10.0 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
RESTORE=true LEARNING_RATE=0.1 LAMBDA=100.0 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
RESTORE=true LEARNING_RATE=0.1 LAMBDA=1000.0 DUMP_FILE="dumps/lambda_magnitude_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py

# RESTORE=false LEARNING_RATE=0.3 LAMBDA=0.1 DUMP_FILE="dumps/training_test.h5" TOTAL_STEPS=3000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.1 LAMBDA=0.1 DUMP_FILE="dumps/training_test.h5" TOTAL_STEPS=20000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.01 LAMBDA=0.1 DUMP_FILE="dumps/training_test.h5" TOTAL_STEPS=20000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.001 LAMBDA=0.1 DUMP_FILE="dumps/training_test.h5" TOTAL_STEPS=20000 ./MNIST_train.py

# RESTORE=false LEARNING_RATE=0.3 LAMBDA=1 DUMP_FILE="dumps/training_test_b.h5" TOTAL_STEPS=3000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.1 LAMBDA=1 DUMP_FILE="dumps/training_test_b.h5" TOTAL_STEPS=20000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.01 LAMBDA=1 DUMP_FILE="dumps/training_test_b.h5" TOTAL_STEPS=20000 ./MNIST_train.py
# RESTORE=true LEARNING_RATE=0.001 LAMBDA=1 DUMP_FILE="dumps/training_test_b.h5" TOTAL_STEPS=20000 ./MNIST_train.py

