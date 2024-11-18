# TODO
def test_with_gb(data, labels) -> None:
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, RocCurveDisplay, auc
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.ensemble import GradientBoostingClassifier

    tprs = []
    fprs = []
    scores = []
    auc_scores = []

    for epoch in range(10):
        print("="*20+f"EPOCH {epoch+1}"+"="*20)
        for i, (train_index, test_index) in enumerate(KFold(10, shuffle=True).split(allDataNorm)):
    #             directory = f"{args.results_directory}/{args.classifier}/epoch_{epoch}"

            x_train = allDataNorm[train_index]
            x_test = allDataNorm[test_index]
            y_train = allDataVectClass[train_index]
            y_test = allDataVectClass[test_index]
            # classifier = LogisticRegression(max_iter=10000)
            classifier = GradientBoostingClassifier(n_estimators=150, max_depth=1)
            # classifier = RandomForestClassifier(n_estimators=100, max_depth=10)


            classifier.fit(x_train, y_train)
            score = classifier.score(x_test, y_test)
            probs = classifier.predict_proba(x_test)
            preds = classifier.predict(x_test)

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

            viz = RocCurveDisplay(fpr=fpr,
                                    tpr=tpr,
                                    roc_auc=auc(fpr, tpr),
                                    estimator_name='---').plot(color="darkorange")
                                    #estimator_name=args.classifier).plot(color="darkorange", plot_chance_level=True)
                # viz.figure_.savefig(f"{directory}/KFold_{i+1}.png")
            viz.figure_.savefig('theFig.png')
            plt.close()
                # precisions.append(precision_score(y_test, preds, labels=[0, 1, 2]))
                # recall.append(recall_score(y_test, preds, labels=[0, 1, 2]))
                # f_score.append(f1_score(y_test, preds, labels=[0, 1, 2]))
            str1 = 'testClass.' + str(epoch+1) + '.' + str(i+1)
            str2 = 'testOut.' + str(epoch+1) + '.' + str(i+1)
            print(str1)
            np.savetxt(str1, y_test, fmt='%f')
            np.savetxt(str2, probs, fmt='%f')

            del classifier
        print()

    max_shape = max([len(a) for a in tprs])
    tprs = [np.append(t, [1 for _ in range(max_shape-len(t))]) for t in tprs]
    fprs = [np.append(f, [1 for _ in range(max_shape-len(f))]) for f in fprs]
    mean_tprs = np.array(tprs)
    mean_tprs = np.mean(mean_tprs, axis=0)
    mean_fprs = np.array(fprs)
    mean_fprs = np.mean(mean_fprs, axis=0)

    mean_auc = 100.0*sum(auc_scores) / len(auc_scores)
    print(mean_auc)