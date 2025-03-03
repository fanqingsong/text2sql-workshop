import polars
import glob
import re
import logging
import tqdm

logging.basicConfig(level = "DEBUG")

print("-------------- main out ------------------")

def main():
    print("-------------- main ------------------")

    connection = "postgresql://postgres:changeme@localhost:5432"

    for csv_file in tqdm.tqdm(glob.glob("../../data/*.csv")):
        print(f'csv_file == {csv_file}')

        logging.info(f'csv_file == {csv_file}')

        table_name = re.search("(?<=\/)[a-z]+(?=\.)", csv_file)[0]
        pdf = polars.read_csv(csv_file, ignore_errors = True)
        pdf.write_database(
            table_name = table_name,
            connection = connection,
            # if_exists = "replace"
        )

if __name__ == "__main__":
    logging.info("Ingesting data into PSQL...")
    main()