start=`date +%s`

# Series of scripts to run for presentation demo. All scripts and data files should be kept locally. 
printf "\nActivating Python environment\n"
source env/bin/activate

#printf "\nRunning extract_data.py\n"
#python /home/david/Documents/ARU/AvalancheProject/demo/extract_data.py

printf "\nRunning transform_ava_coords.py\n"
python /home/david/Documents/ARU/AvalancheProject/demo/transform_ava_coords.py

printf "\nStarting PSQL scripts.\n"
printf "\nBuilding Database: 'avalanche'.\n"
psql -d avalanche -U david -f BuildDB.sql

printf "\nDetermining nearest weather station to avalanche event.\n"
psql -d avalanche -U david -f ./find_nearest_station.sql

#printf "\nExtracting weather data for avalanches and generating dull data.\n"
#python /home/david/Documents/ARU/AvalancheProject/demo/extract_data.py

printf "\nProcessing weather data.\n"
python /home/david/Documents/ARU/AvalancheProject/demo/transformWeather.py

printf "\nCleaning data and building datasets.\n"
python /home/david/Documents/ARU/AvalancheProject/demo/transformFinalData.py

printf "\nRunning Random Forest Classifier.\n"
python /home/david/Documents/ARU/AvalancheProject/demo/modelling/demo_pipe.py


end=`date +%s`

runtime=$(($end-$start))
printf "\nTotal time taken: $runtime s\n"