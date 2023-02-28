import argparse
from filter_stations import filterStations

def main():
    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--address', required=True, help='Address to filter stations by')
    parser.add_argument('--distance', type=int, default=100, help='Distance in kilometers to search from address')
    parser.add_argument('--start-date', help='Filter by start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Filter by end date (YYYY-MM-DD)')
    parser.add_argument('--csv-file', default='KEcheck3.csv', help='Path to input CSV file')
    args = parser.parse_args()

    filtered_stations = filterStations.filterStations(args.address, args.distance, args.start_date, args.end_date, args.csv_file)
    print(filtered_stations)

if __name__ == '__main__':
    main()
