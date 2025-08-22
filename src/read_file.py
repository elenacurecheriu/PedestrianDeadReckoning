import pandas as pd

def read_file(dir_name, file_name):
    file_path = f"{dir_name}/{file_name}.csv"
    df = pd.read_csv(file_path, header=None)
    df.columns = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "heading", "db", "frequency"]
    df = df.iloc[:, :-2]
    # remove header
    df = df[df['ax'] != 'ax']
    df[['ax', 'ay', 'az']] = df[['ax', 'ay', 'az']].astype(float)
    df['svm'] = (df[['ax', 'ay', 'az']] ** 2).sum(axis=1) ** 0.5
    return df
        

if __name__ == "__main__":
    read_file("../data", "log_1")
