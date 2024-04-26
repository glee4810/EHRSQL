import argparse

def parse_arguments():
    """Set up and return command line arguments."""
    parser = argparse.ArgumentParser(description="Process and transform EHR data.")

    parser.add_argument('--data_dir', required=True, type=str, help='Directory containing CSV files.')
    parser.add_argument('--db_name', required=True, type=str, choices=['mimic_iii', 'eicu', 'mimic_iv'], help='Database name: mimic_iii, eicu, or mimic_iv.')
    parser.add_argument('--out_dir', default='../dataset/ehrsql', type=str, help='Output directory for files.')

    parser.add_argument('--sample_icu_patient_only', action='store_true', help='Sample only patients who went to the ICU.')
    parser.add_argument('--num_patient', default=1000, type=int, help='Number of patients to process.')

    parser.add_argument('--deid', action='store_true', help='Enable de-identification of data.')
    parser.add_argument('--timeshift', action='store_true', help='Enable time shift of data records.')
    parser.add_argument('--start_year', type=int, help='Starting year for sampling when timeshifting.')
    parser.add_argument('--time_span', type=int, help='Time span starting from start_year for timeshifting.')
    parser.add_argument('--cur_patient_ratio', default=0.0, type=float, help='Ratio of current patients to process.')
    parser.add_argument('--current_time', type=str, help='Current time threshold for data inclusion.')

    return parser.parse_args()


def process_database(args):
    """Process the specific database based on db_name argument."""
    if args.db_name == 'mimic_iii':
        from preprocess_db_mimic_iii import Build_MIMIC_III as DB_Builder
    elif args.db_name == 'mimic_iv':
        from preprocess_db_mimic_iv import Build_MIMIC_IV as DB_Builder
    elif args.db_name == 'eicu':
        from preprocess_db_eicu import Build_eICU as DB_Builder
    else:
        raise ValueError(f"Unsupported database name: {args.db_name}")

    db_builder = DB_Builder(data_dir=args.data_dir, out_dir=args.out_dir, db_name=args.db_name, 
                            num_patient=args.num_patient, sample_icu_patient_only=args.sample_icu_patient_only,
                            deid=args.deid, timeshift=args.timeshift, 
                            start_year=args.start_year, time_span=args.time_span, 
                            cur_patient_ratio=args.cur_patient_ratio, current_time=args.current_time)

    db_builder.generate_db()

def main(args):
    """Main function to process the EHR data based on the provided arguments."""
    if args.timeshift:
        required_fields = ['start_year', 'time_span', 'current_time']
        if not all(getattr(args, field) is not None for field in required_fields):
            raise ValueError('All time shifting fields must be specified when --timeshift is enabled.')

    process_database(args)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    print('Processing complete!')
