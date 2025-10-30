## Building with Pallas
We recommend using `pkg-config` to get the correct flags when building with Pallas.
```bash
gcc youfile.c $(pkg-config --libs pallas)  $(pkg-config --cflags pallas) 
```

You can also use CMake:
```cmake
pkg_check_modules(PALLAS pallas REQUIRED)
target_link_libraries(my-lib PRIVATE ${PALLAS_LIBRARIES})
target_link_directories(my-lib PRIVATE ${PALLAS_LIBRARY_DIRS})
```

## Using Pallas
This section assumes you've read ["Learning about the Pallas trace format"](01-presentation).
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
```
