from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
import numpy as np
from Goowe import Goowe
import pandas as pd
import time
from scipy.io import arff

# Prepare the data stream

# data_1 = np.load('email_data_numpy.npy' ,  allow_pickle=True)
# dataset = np.load('email_data_numpy.npy' , allow_pickle=True)

# np.savetxt("email_data_csv.csv", dataset, delimiter=",")
# asdasd


# df = pd.read_csv(r"turkish_news_modified_abrupt.csv" )
# df = pd.read_csv(r"turkish_news_modified_gradual.csv" )
# /home/sepehr/Downloads/SEPEHR/EBLS/SIGIR2020/EBLS/IEEEAccess_LED_00_07.csv
f = open("data.txt", "a")

for o in range(1,2):


    accuracyArr = np.zeros(7)
    if o == 0:
        f.write("-------------------elec------------------")
        p_tot = np.zeros(1)
        p_total_tot = np.zeros(1)
        ps = np.zeros(7)
        pts = np.zeros(7)
    elif o == 1:
        f.write("\n-------------------rialto------------------")
        p_tot = np.zeros(9)
        p_total_tot = np.zeros(9)
        ps = np.zeros((7,9))
        pts = np.zeros((7,9))
    elif o == 2:
        f.write("\n-------------------rbf2------------------")
        p_tot = np.zeros(1)
        p_total_tot = np.zeros(1)
        ps = np.zeros(7)
        pts = np.zeros(7)
    elif o == 3:
        f.write("\n-------------------rbf4------------------")
        p_tot = np.zeros(3)
        p_total_tot = np.zeros(3)
        ps = np.zeros((7,3))
        pts = np.zeros((7,3))
    elif o == 4:
        f.write("\n-------------------rbf8------------------")
        p_tot = np.zeros(7)
        p_total_tot = np.zeros(7)
        ps = np.zeros((7,7))
        pts = np.zeros((7,7))
    elif o == 5:
        f.write("\n-------------------rbf16------------------")
        p_tot = np.zeros(15)
        p_total_tot = np.zeros(15)
        ps = np.zeros((7,15))
        pts = np.zeros((7,15))
    elif o == 6:
        f.write("\n-------------------covtype------------------")
        p_tot = np.zeros(6)
        p_total_tot = np.zeros(6)
        ps = np.zeros((7,6))
        pts = np.zeros((7,6))

    for j in range(1, 8):
        print("j = ", j)

        if o == 0:
            stream = FileStream(r"elec.csv")
            num_classes = 2  # len(stream.get_target_values())
            target_values = [0, 1]  # stream.get_target_values()
        elif o == 1:
            stream = FileStream(r"rialto.csv")
            num_classes = 10  # len(stream.get_target_values())
            target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # stream.get_target_values()
        elif o == 2:
            stream = RandomRBFGenerator(n_classes=2, n_features=20)
            num_classes = 2  # len(stream.get_target_values())
            target_values = [0, 1]  # stream.get_target_values()
        elif o == 3:
            stream = RandomRBFGenerator(n_classes=4, n_features=20)
            num_classes = 4  # len(stream.get_target_values())
            target_values = [0, 1, 2, 3]  # stream.get_target_values()
        elif o == 4:
            stream = RandomRBFGenerator(n_classes=8, n_features=20)
            num_classes = 8  # len(stream.get_target_values())
            target_values = [0, 1, 2, 3, 4, 5, 6, 7]  # stream.get_target_values()
        elif o == 5:
            stream = RandomRBFGenerator(n_classes=16, n_features=20)
            num_classes = 16  # len(stream.get_target_values())
            target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # stream.get_target_values()
        else:
            stream = FileStream(r"covtype.csv")
            num_classes = 7  # len(stream.get_target_values())
            target_values = [0, 1, 2, 3, 4, 5, 6]  # stream.get_target_values()

        # stream = RandomRBFGenerator(model_random_state=3, n_classes=16)

        stream.prepare_for_use()

        num_features = stream.n_features
        num_targets = 1  # stream.n_targets

        N_MAX_CLASSIFIERS = 2 ** j
        CHUNK_SIZE = 500  # User-specified
        WINDOW_SIZE = 100  # User-specified

        # Initialize the ensemble
        goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
                      chunk_size=CHUNK_SIZE,
                      window_size=WINDOW_SIZE,
                      logging=False)
        goowe.prepare_post_analysis_req(num_features, num_targets, num_classes, target_values)

        # For the first chunk, there is no prediction.

        X_init, y_init = stream.next_sample(CHUNK_SIZE)
        # print(X_init)
        # print(y_init)

        goowe.partial_fit(X_init, y_init)

        accuracy = 0.0
        total = 0.0
        true_predictions = 0.0

        for i in range(CHUNK_SIZE):
            total += 1
            cur = stream.next_sample()
            X, y = cur[0], cur[1]
            # print(X)
            preds = goowe.predict(X)

            true_predictions += np.sum(preds == y)
            accuracy = true_predictions / total
            # print('\tData instance: {} - Accuracy: {}'.format(total, accuracy))

            goowe.partial_fit(X, y)

        # Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
        start = time.time()
        acc_eac_100 = []
        time_each_100 = []

        k = 0
        while stream.has_more_samples() and k < 100000:
            k += 1
            total += 1
            cur = stream.next_sample()
            X, y = cur[0], cur[1]

            preds = goowe.predict(X)  # Test
            true_predictions += np.sum(preds == y)
            accuracy = true_predictions / total
            # print('\tData instance: {} - Accuracy: {}'.format(int(total), round(accuracy * 100.0, 3)))
            goowe.partial_fit(X, y)  # Then train
            # print(total)

            if (total % 20 == 0):
                # time.sleep(2)
                overall_time = time.time() - start
                acc_eac_100.append(accuracy)
                time_each_100.append(overall_time)

        ############################ UPDATE: To calculate probabilities
        accuracyArr[j - 1] = accuracy
        ps[j - 1] = goowe.probabilityCalculator.p_array
        pts[j - 1] = goowe.probabilityCalculator.p_total
        print("p_arr: ", goowe.probabilityCalculator.p_array)
        f.write("\np_arr: " +str( goowe.probabilityCalculator.p_array))
        print("p_total: ", goowe.probabilityCalculator.p_total)
        f.write("\np_total: " +str( goowe.probabilityCalculator.p_total))
        p_tot += goowe.probabilityCalculator.p_array
        p_total_tot += goowe.probabilityCalculator.p_total
        ############################

        print(len(acc_eac_100))
        print(time.time() - start)
        # np.save("usenet_Goowe_acc" ,acc_eac_100)
        # np.save("usenet_Goowe_time" ,time_each_100)

    print("accuracyArr: ", accuracyArr)
    print("ps: ", ps)
    print("pts: ", pts)
    print("total p_arr: ", p_tot)
    print("total p-tot: ", p_total_tot)
    f.write("\naccuracyArr: " + str(accuracyArr))
    f.write("\nps: " + str(ps))
    f.write("\npts: " + str(pts))
    f.write("\ntotal p_arr: " + str(p_tot))
    f.write("\ntotal p-tot: " + str(p_total_tot))
    # numClassifiers = goowe.probabilityCalculator.calculateNumberOfClassifiers(p_tot, p_total_tot)
    # print(" numClassifiers: ", numClassifiers)
    print("average accuracy: ", sum(accuracyArr) / len(accuracyArr))
    # f.write("\naverage numClassifiers: " + str(numClassifiers))
    f.write("\naverage accuracy: " + str(sum(accuracyArr) / len(accuracyArr)))


