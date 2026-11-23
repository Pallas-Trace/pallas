# Using Pallas

This section assumes you've read the [Pallas format](01-format.md).
A global rule when using the Pallas library is to ***never copy an object***.
Always use pointers and references when possible: the memory management is made by the Pallas library internally,
you should ***not*** have to deal with it.

You should always start by creating a single `GlobalArchive`.
Again, this object should never be duplicated, or you will risk issues with duplications.
You can then begin defining your LocationGroups for your Archives, or some global definitions.
Once an Archive has been created, you can create your Locations for your Threads.
```cpp
#include <pallas/pallas_write.h>
int main() {
      GlobalArchive globalArchive("<my_trace_name>", "<file_name>");
      // This will create the <my_trace_name> folder, and the main trace will be called <file_name>
      StringRef processName = globalArchive.registerString("Main process");
      globalArchive.defineLocationGroup(0, processName, -1);
      // The first LocationGroup has no parent, hence -1
      Archive mainProcess(globalArchive, 0);
      // On other processes where globalArchive is not accessible, one can use the other constructor:
      // Archive mainProcess("<my_trace_name>", mpi_rank);
      StringRef threadName = archive.registerString("Main Thread");
      mainProcess.defineLocation(threadID, threadName, 0);
      // threadID is whatever you want
      // 0 is the parent's ID
      ThreadWriter threadWriter(archive, threadID);
      // This automatically creates a Thread in the Archive, and handles the memory of said Thread.
}
```
Once all that is done, you can begin logging your events.
Before logging an event, you might have to register Strings, Regions, Definitions, etc.
It is recommended to do that before the execution starts, otherwise it might interfere with your performance (although it's most likely negligible).
```cpp
int main() {
    /** Previous stuff ... **/
    StringRef functionName = globalArchive.registerString("Common Function");
    RegionRef commonFunction = myCustomFunctionId;
    globalArchive.addRegion(commonFunction, functionName);
    // This register a region to be used globally.
    // You can do that same for local regions
    // Be aware that any local definition will overwite global definitions
    // So be carefull or you'll end up with traces that make no sense.
    pallas_record_<event_type>(&threadWriter, args...);
}
```

Once the execution is finished, you need to write the Threads, Archives and GlobalArchive manually.
```cpp
int main() {
    /** End of Execution **/
    threadWriter.threadClose();
    mainProcess.store();
    globalArchive.store();
}
```
### Storing custom data
There are two ways a user might want to store custom data in Pallas:
either in the GlobalArchive/Archive (for information about the execution, for example),
or in an Event (for information concerning a specific logged event).
The former can be done by using the `AdditionalData<T>` template, which takes a pointer and a writer function
(allowing you to create complex ways of writing your custom data):
```cpp
// Define a custom structure that holds the data
struct ExampleData {
    std::string name;
    size_t random_number;
};
// Define a custom function that writes the data to a file
size_t write_example_data(example_data* d, FILE* file) {
    return fwrite(d, sizeof(example_data), 1, file);
}

// Then adding the example data to the archive would look like this:
void record_example_data(pallas::Archive& archive) {
      ExampleData data{"TestData", 42};
      AdditionalContent<ExampleData> content{&data, &write_example_data};
      archive.add_content(&content);
}
```

Custom data in Events cannot currently be done at runtime.
However, it is quite easy to add your own custom events.
