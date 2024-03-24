import functions as fn


def main():
    df = fn.readDataset()
    fn.showDfMetrics(df)


if __name__ == "__main__":
    main()
