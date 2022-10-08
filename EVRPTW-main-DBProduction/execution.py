from msilib.schema import Error
import os, csv
from time import time


db_Output = open('./DB-Output.csv', 'w', newline='')
writer = csv.writer(db_Output)
writer.writerow(["Instance's Name","Seed","Initial Solution","OFIS","Moves","OFFS","OF_Diff","Exe_Time_d-r","Avg_Battery_Status","Avg_SoC","Avg_Num_Charge",
                 "Avg_Vehicle_Capacity","Avg_Customer_Demand","Num_Vehicles","Avg_Service_Time","Avg_Customer_TimeWindow","Var_Customer_TimeWindow",
                 "Avg_Customer_customer_min_dist","Var_Customer_customer_min_dist","Avg_Customer_station_min_dist","Var_Customer_station_min_dist",
                 "Avg_Customer_deposit_dist","Var_Customer_deposit_dist","CounterD_R","CounterD_Rlast"])
db_Output.close()

counterinit=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

fC = open('./Counter.csv', 'w')
writer = csv.writer(fC)
writer.writerow(counterinit)
fC.close()

fCL = open('./Counterlast.csv', 'w')
writer = csv.writer(fCL)
writer.writerow(counterinit)
fCL.close()

# Re-initialize ./etc/settings.json
def initialize_file_settings():
    new_file = """{
        "unit_energy_cost" : 0.4,
        "driver_wage" : 1,
        "fixed_vehicle_acquisition" : 1200,
        "overtime_cost_numerator" : 11,
        "overtime_cost_denominator" : 6,
        "rho_low" : 0.3,
        "rho_high" : 0.7,
        "instance_file_name" : "c101_21_25.txt",
        "service_time_generation_type" : "basic",
        "basic_service_time" : {
            "R" : { "low" : 8,
                    "high" : 12},
            "RC" : { "low" : 8,
                    "high" : 12},
            "C" : { "low" : 70,
                    "high" : 1100}
        }
    }"""

    with open('./etc/settings.json', 'w') as file:
        file.write(new_file)

def one_hour_running_code():
    # Makes the code running for an hour
    start_time = time()
    
    # Save all instances from ./data
    files_list = []
    for file in os.listdir('./data'):
        files_list.append(file)

    # Makes each instance running 10 times for 10 iterations
    for i in range(len(files_list)):
        if (time() - start_time) > 3600: break
        for j in range(10):
            try:
                os.system("python main.py")
            except Exception as e:
                pass
            
        fileR = open('./etc/settings.json', 'r')
        filedata = fileR.read()
        fileR.close()

        filedata = filedata.replace(files_list[i], files_list[i+1])

        fileW = open('./etc/settings.json', 'w')
        fileW.write(filedata)
        fileW.close()

if __name__ == "__main__":

    initialize_file_settings()

    one_hour_running_code()

    # Changing seed from '123' to '42'
    fileR_main = open('./main.py', 'r')
    filedata_main = fileR_main.read()
    fileR_main.close()

    filedata_main = filedata_main.replace('123', '42')

    fileW_main = open('./main.py', 'w')
    fileW_main.write(filedata_main)
    fileW_main.close()

    initialize_file_settings()

    one_hour_running_code()
