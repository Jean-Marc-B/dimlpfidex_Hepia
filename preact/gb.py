import os
import shutil
import numpy as np
import pandas as pd
import src.data_helper as dh
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, RocCurveDisplay, auc


def delete_all_content_of_dir(dirpath: str) -> None:
    for filename in os.listdir(dirpath):
        file_path = os.path.join(dirpath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def test_with_gb(data: pd.DataFrame, labels: pd.Series) -> None:
    tprs = []
    fprs = []
    scores = []
    auc_scores = []

    directory = "gb_tests/"
    delete_all_content_of_dir(directory)

    for epoch in range(10):
        print("=" * 20 + f"EPOCH {epoch+1}" + "=" * 20)
        subdir = f"epoch_{epoch}/"
        os.mkdir(directory + subdir)

        for i, (train_index, test_index) in enumerate(
            KFold(10, shuffle=True).split(data, labels)
        ):

            x_train = data.iloc[train_index]
            x_test = data.iloc[test_index]
            y_train = labels.iloc[train_index]
            y_test = labels.iloc[test_index]
            # classifier = LogisticRegression(max_iter=10000)
            classifier = GradientBoostingClassifier(n_estimators=150, max_depth=1)
            # classifier = RandomForestClassifier(n_estimators=100, max_depth=10)

            classifier.fit(x_train, y_train)
            score = classifier.score(x_test, y_test)
            probs = classifier.predict_proba(x_test)
            # preds = classifier.predict(x_test)

            label_binarizer = LabelBinarizer().fit(y_train)
            y_one_hot_test = label_binarizer.transform(y_test)

            print(f"KFold {i + 1} -> SCORE : {score}")
            scores.append(score)
            # fpr, tpr, tresholds = roc_curve(y_one_hot_test[:, 1], probs[:, 1])
            fpr, tpr, tresholds = roc_curve(y_one_hot_test, probs[:, 1])
            # fpr, tpr, tresholds = roc_curve(y_one_hot_test[:, 0], probs[:, 0])
            # fpr, tpr, tresholds = roc_curve(y_one_hot_test[:, 2], probs[:, 2])
            fprs.append(fpr)
            tprs.append(tpr)
            auc_scores.append(auc(fpr, tpr))

            viz = RocCurveDisplay(
                fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name="---"
            ).plot(color="darkorange")
            # estimator_name=args.classifier).plot(color="darkorange", plot_chance_level=True)
            # viz.figure_.savefig(f"{directory}/KFold_{i+1}.png")
            viz.figure_.savefig(directory + subdir + "AUC.png")
            plt.close()
            # precisions.append(precision_score(y_test, preds, labels=[0, 1, 2]))
            # recall.append(recall_score(y_test, preds, labels=[0, 1, 2]))
            # f_score.append(f1_score(y_test, preds, labels=[0, 1, 2]))
            str1 = "testClass." + str(epoch + 1) + "." + str(i + 1)
            str2 = "testOut." + str(epoch + 1) + "." + str(i + 1)
            np.savetxt(directory + subdir + str1, y_test, fmt="%f")
            np.savetxt(directory + subdir + str2, probs, fmt="%f")

            del classifier
        print()

    max_shape = max([len(a) for a in tprs])
    tprs = [np.append(t, [1 for _ in range(max_shape - len(t))]) for t in tprs]
    fprs = [np.append(f, [1 for _ in range(max_shape - len(f))]) for f in fprs]
    # mean_tprs = np.array(tprs)
    # mean_tprs = np.mean(mean_tprs, axis=0)
    # mean_fprs = np.array(fprs)
    # mean_fprs = np.mean(mean_fprs, axis=0)

    mean_auc = 100.0 * sum(auc_scores) / len(auc_scores)
    print(f"MEAN AUC: {mean_auc:.3f}")


if __name__ == "__main__":
    data, labels = dh.obtain_data("dataset/clinical_complete_rev1.csv")
    data, labels = dh.filter_clinical(data, labels)

    # added BMI column
    data = data.assign(BMI=lambda x: round(x.WEIGHT / (x.HEIGHT / 100.0) ** 2, 3))

    test_with_gb(data, labels)
