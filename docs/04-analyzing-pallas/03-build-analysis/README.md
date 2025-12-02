---
title: Build you own program
---
# Build Your Own Analysis Programs

Developer guide for creating custom analysis programs for Pallas traces.

TODO

## Python API

Pallas provides a Python API for reading Pallas traces. To install it,
enable the `ENABLE_PYTHON` option when building Pallas.

You can then import pallas in your python script as follows

```
import pallas_trace as pallas
trace_name="bt.C.64_trace/eztrace_log.pallas"
trace = pallas.open_trace(trace_name)
```

### Iterating over threads

```
for process in trace.archives:
  for thread in process.threads:
    print(thread.id)

```

### Printing a thread's event list

```
def print_pallas_object(obj: pallas.Sequence| pallas.Loop | pallas.Event, index: int):
    return
    match type(obj):
        case pallas.Sequence:
            print(f\"{obj.timestamps[index]/1e9}\\t{obj.id}\")
        case pallas.Event:
            print(f\"{obj.timestamps[index]/1e9}\\t{obj.id}\")

def print_sequence(s: pallas.Sequence, counter: dict):
    for temp in s.content:
        if temp.id not in counter:
            counter[temp.id] = 0
        print_pallas_object(temp, counter[temp.id])
        counter[temp.id] +=1
        match type(temp):
            case pallas.Sequence:
                print_sequence(temp, counter)
            case pallas.Loop:
                for loop in range(temp.nb_iterations):
                    if temp.sequence.id not in counter:
                        counter[temp.sequence.id] = 0
                    print_pallas_object(temp.sequence, counter[temp.sequence.id])
                    print_sequence(temp.sequence, counter)
                    counter[temp.sequence.id] += 1

def print_thread(thread: pallas.Thread):
    # we need a dictionnary to iterate over pallas sequences
    counter = {}
    print_sequence(thread.sequences[0], counter)
```

### Printing the list of functions in a thread
```
main_thread = trace.archives[0].threads[0]
for s in main_thread.sequences:
    print(f"{s.id}\t{s.guessName()}\t{s.min_duration / 1e9}\t{s.max_duration/1e9}\t{s.mean_duration/1e9}\t{s.n_iterations}")
```

Expected result:
```
S0	Sequence_0	48.75305308	48.75305308	48.75305308	1
S1	MPI_Comm_dup	0.000157332	0.002357128	0.00125723	2
S2	mpi_bcast_	2.16e-06	0.000119776	3.9902e-05	4
S3	mpi_bcast_	6.18e-06	6.18e-06	6.18e-06	1
S4	mpi_bcast_	5.392e-06	5.392e-06	5.392e-06	1
S5	mpi_irecv_	1.114e-05	8.8124e-05	2.0297e-05	201
S6	mpi_irecv_	1.868e-06	8.5104e-05	4.471e-06	201
S7	mpi_irecv_	1.6e-06	5.6104e-05	4.595e-06	201
S8	mpi_irecv_	2.592e-06	1.9372e-05	4.702e-06	201
S9	mpi_irecv_	1.208e-06	4.4652e-05	3.611e-06	201
S10	mpi_irecv_	1.908e-06	2.524e-05	3.68e-06	201
S11	mpi_isend_	6.42e-06	3.5692e-05	1.2836e-05	201
S12	mpi_isend_	3.572e-06	1.826e-05	6.658e-06	201
S13	mpi_isend_	2.592e-06	1.1728e-05	5.584e-06	201
S14	mpi_isend_	2.692e-06	8.692e-06	5.017e-06	201
S15	mpi_isend_	2.688e-06	0.000181224	6.385e-06	201
S16	mpi_isend_	3.032e-06	7.3592e-05	5.483e-06	201
S17	mpi_waitall_	0.004213872	0.009983292	0.007709838	201
```

### Creating a communication matrix
```
# Creating a communication matrix
matrix = np.zeros((len(trace.archives), len(trace.archives)))

for sender, archive in enumerate(trace.archives):
    for thread in archive.threads:
        for event in thread.get_events_from_record(pallas.Record.MPI_ISEND):
            data = event.data
            matrix[sender][data['receiver']] += data['msgLength']

plt.matshow(matrix)"
```

### Plotting the distribution of sequence duration

```
# Plotting an histogram to see the time distribution of a certain sequence
selected_sequence = thread.sequences[20]
plt.hist(selected_sequence.durations.as_numpy_array() / 1e9)"
```
