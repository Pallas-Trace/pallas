import pallas
import pandas as pd
import numpy as np
import sys
import datetime
print("Libraries loaded !")
def create_empty_df():
    df=['Thread', 'Function', 'Start', 'Finish', 'Duration']
    df = pd.DataFrame({"Thread":pd.Series(dtype='str'),
                       "Function":pd.Series(dtype='str'),
                       "Start":pd.Series(dtype='timedelta64[ns]'),
                       "Finish":pd.Series(dtype='timedelta64[ns]'),
                       "Duration":pd.Series(dtype='int64')})
    return df

def read_trace_pallas(filename):
    trace=pallas.open_trace(filename)
    df = create_empty_df()
    dataframes_list=[]
    for archive in trace.archives:
        for thread in archive.threads:
            for seq in thread.sequences:
                dataset = create_empty_df()
                dataset["Start"]=np.array(seq.timestamps)
                dataset["Duration"]=np.array(seq.durations)
                dataset["Finish"]=dataset["Start"]+dataset["Duration"]
                dataset["Thread"]=trace.locations[thread.id].name
                dataset["Function"]=seq.guessName(thread)
                dataframes_list.append(dataset)

    df=pd.concat(dataframes_list, axis=0, ignore_index=True) 
    df=df.sort_values(["Start", "Finish"], ascending=[True, False])
    df=df.reset_index(drop=True)
    return df

def compute_depth(df):
    t1=datetime.datetime.now()

    threads=sorted(df["Thread"].unique())

    # Depth is a dict of int ("Thread"->cur_depth)
    depth={}
    # Finish_ts is a dict of array(int) ("Thread"->[finish_ts_frame_0, finish_ts_frame_1,...]
    finish_ts={}
    indexes={}
    
    for index, row in df.iterrows():
        t=row["Thread"]
        if not t in depth:
            # First event for this thread
            print(f"first event for {t} at ts {row['Start']}")
            depth[t]=0
            finish_ts[t]=[row["Finish"]]
            indexes[t]=[index]
        else:
            start_ts=row["Start"]

            while(start_ts >= finish_ts[t][-1]):
                # remove the last element of the list
                del finish_ts[t][-1]
                del indexes[t][-1]
                depth[t]=depth[t]-1

            depth[t] = depth[t] + 1
            print(f"{row['Thread']} {row['Function']}\t [{row['Start']}-{row['Finish']}]\t is in seq [?-{finish_ts[t][-1]}]")

            if row["Finish"] > finish_ts[t][-1]:
                print("Error !")
                print(df.loc[[index]])
                print(df.loc[[indexes[t][-1]]])
                raise Exception()
            finish_ts[t].append(row["Finish"])
            indexes[t].append(index)

        df.at[index, "Depth"] = depth[t]
    t2=datetime.datetime.now()
    d=t2-t1
    print("Compute depth took "+str(d))
    return df

filename=sys.argv[1]
df=read_trace_pallas(filename)
df=compute_depth(df)
