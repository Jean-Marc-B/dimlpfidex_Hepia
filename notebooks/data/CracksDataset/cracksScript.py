from dimlpfidex import fidex

def main():
    fidex.fidexGloRules('--train_data_file trainData.txt --train_pred_file predTrain.out --train_class_file trainClass.txt --weights_file weights.wts --nb_quant_levels 100 --global_rules_outfile globalRules.txt --heuristic 1 --nb_attributes 4096 --nb_classes 2 --max_iterations 25 --min_covering 2 --dropout_dim 0.9 --dropout_hyp 0.9 --console_file fidexGloRulesResult.txt --root_folder notebooks/data/CracksDataset --nb_threads 8')
if __name__ == '__main__':
    main()
