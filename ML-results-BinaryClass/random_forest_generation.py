import pandas, joblib, csv, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

moves_list = ["RandomDestroyStation","LongestWaitingTimeDestroyStation","DeterministicBestRepairStation","ProbabilisticBestRepairStation","GreedyDestroyCustomer",
                "WorstDistanceDestroyCustomer","WorstTimeDestroyCustomer","RandomRouteDestroyCustomer","ZoneDestroyCustomer","DemandBasedDestroyCustomer",
                "TimeBasedDestroyCustomer","ProximityBasedDestroyCustomer","ShawDestroyCustomer","GreedyRouteRemoval","ProbabilisticWorstRemovalCustomer","GreedyRepairCustomer"]
                # ProbabilisticGreedyRepairCustomer, ProbabilisticGreedyConfidenceRepairCustomer

# Import dataset
df = pandas.read_csv('../EVRPTW-main-DBProduction/DB-Output.csv')

# Balancing samples
def write_db_bal(g_limit, b_limit, move): 
    g_limit3 = g_limit * 3
    print("g_limit: " + str(g_limit))
    print("b_limit: " + str(b_limit))
    g_count, b_count = 0,0

    with open("./DB-Output-" + move + "-Bal1.csv", 'r') as olddb:
        readerolddb = csv.reader(olddb)

        newdb = open("./DB-Output-" + move + "-Bal.csv", 'w', newline='')
        writernewdb = csv.writer(newdb)
        writernewdb.writerow(next(readerolddb))

        for row in readerolddb:
            if (row[-1] == "Good" and g_count <= g_limit3):
                writernewdb.writerow(row)
                writernewdb.writerow(row)
                writernewdb.writerow(row)
                g_count += 1
            elif (row[-1] == "Bad" and b_count <= g_limit3):
                writernewdb.writerow(row)
                b_count += 1
    
        newdb.close()
    
    os.system("del .\\DB-Output-" + move + "-Bal1.csv")


if __name__ == "__main__":

    for move in moves_list:

        # Filtering dataset foreach moves, removing useless fields and adding column label
        df_move_undropped = df[df["Moves"].str.contains(move)]
        df_moveX = df_move_undropped.drop(['Seed','CounterD_R','CounterD_Rlast','Initial Solution',"Instance's Name",'Moves'], axis=1)
        df_moveX["DIFF4CLAS"] = ["" for i in range(len(df_moveX.index))]
        df_moveX['DIFF4CLAS'].mask( df_moveX['OF_Diff'].apply(lambda x: float(x)) <= 1 ,'Bad', inplace=True)
        df_moveX['DIFF4CLAS'].mask( df_moveX['OF_Diff'].apply(lambda x: float(x)) > 1 ,'Good', inplace=True)
        df_move = df_moveX.drop(['OF_Diff'], axis=1)

        df_move.to_csv("./DB-Output-" + move + "-Bal1.csv", sep=',', index=False)

        write_db_bal(df_move["DIFF4CLAS"].value_counts()["Good"], df_move["DIFF4CLAS"].value_counts()["Bad"], move)
        
        df_move_bal = pandas.read_csv("./DB-Output-" + move + "-Bal.csv")

        # Splitting train and test set
        X = df_move_bal.iloc[:, 0:18].values    # 19 = num tot classses
        y = df_move_bal.iloc[:, 18].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Training and Testing phase
        rf_classifier = RandomForestClassifier(n_estimators=10, criterion='gini', random_state=0, max_depth=15)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        print("++++++++++++++++++++++++++++++++++++\n")
        print("Move: " + move)
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        print("++++++++++++++++++++++++++++++++++++\n")

        # Saving model
        joblib.dump(rf_classifier, "./random_forest_" + move + ".joblib")

        os.system("del .\\DB-Output-" + move + "-Bal.csv")
