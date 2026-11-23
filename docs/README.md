## 🐱 What is Pallas?

Pallas is a state-of-the-art tracing format designed to help developers
and researchers analyze the performance of parallel applications.
Whether you're working with MPI, OpenMP, CUDA, or other parallel paradigms,
Pallas is able to trace and analyze your application's behavior, even at larger scales.
Pallas is a library offering a OTF2 interface, so that it can be used with any tracing tool relying on the OTF2 format.
In this guide, we rely on [EZTrace](https://gitlab.com/eztrace/eztrace) to generate Pallas traces.
However, Pallas can be used without EZTrace, and we will not give a guide to EZTrace either.


![logo](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimages5.fanpop.com%2Fimage%2Fphotos%2F27700000%2FPallas-Cat-pallas-cats-27745375-1600-1067.jpg&f=1&nofb=1&ipt=a060f207d882952b617bcdcbc59d287fc008cc4139d9a9e19f02f3797ab03db6 ':size=400')

_The Pallas cat is just like our trace format: it looks big, but it's actually just very fluffy !_

---

## 🚀 Getting Started

New to Pallas? Start here:

1. **🏃‍♂️ Quick Start Guide**: [Get running in 5 minutes](01-quick-start.md)
2. **🔧 Pallas documentation**: [The Pallas trace format](02-pallas/README.md)
3. **🔍 Scalable trace analysis** with [Pallas native tools](04-analyzing-pallas/README.md)
4. **📊 Trace visualization** at scale with [Blup](05-visualizing-blup/README.md)

---

## 📚 Table of Contents  

### 🏃‍♂️ [Quick Start](01-quick-start.md)
Quickly get up and running with Pallas and EZTrace


### 📈 [Pallas Documentation](02-pallas/README.md)
Learn how to use the **Pallas trace format** to speed up your analyses:
- 🎭 [Pallas Format Overview](02-pallas/01-format.md): All you need to understand Pallas
- 🔧 [Building and Installing Pallas](02-pallas/02-building.md): A detailed guide on all the options for Pallas
- ✍️ [Using Pallas](02-pallas/03-using.md): Write your own traces and store custom data
- Tracing your applications by yourself:
    - ⚡ [Multithreaded Application Tracing (OpenMP)](03-tracing-examples/01-multithread.md)
    - 🌐 [Multiprocess Application Tracing (MPI)](03-tracing-examples/02-multiprocess.md)
    - 📚 [Custom Library Tracing](03-tracing-examples/03-custom-library.md)

### 🔍 [Scalable Trace Analysis with Pallas Traces](04-analyzing-pallas/README.md)
Perform scalable trace analysis using the **Pallas trace format** :
- 🎯 [Analysis Fundamentals](04-analyzing-pallas/01-generalities/README.md)
- 🛠️ [Native Analysis Tools](04-analyzing-pallas/02-native-analysis/README.md)
- 🏗️ [Custom Analysis Programs](04-analyzing-pallas/03-build-analysis/README.md)

### 📊 [Scalable Visualization of Pallas Traces with Blup](05-visualizing-blup/README.md)
Scalable trace visualization relying on the **Pallas API** using the **Blup** tool:
- 🎨 [Blup Overview](05-visualizing-blup/01-generalities/README.md)
- 📂 [Opening Traces](05-visualizing-blup/02-open-trace/README.md)
- ✨ [Advanced Features](05-visualizing-blup/03-other-functionalities/README.md)

### 📖 [API Reference](06-api-reference/README.md)
Complete technical reference for developers and advanced users.

---

## 🎯 Popular Use Cases

- **🌐 MPI Applications**: Trace communication patterns and identify bottlenecks
- **⚡ Hybrid Applications**: Analyze MPI+OpenMP or MPI+CUDA applications
- **🔧 Custom Libraries**: Instrument your own code for detailed analysis
- **📊 Performance Optimization**: Identify hotspots and limitation factors of your applications

---

## 🤝 Contributing

This documentation is part of the Pallas project. We welcome contributions!

- 📝 Found a bug? Want to improve docs? Submit a pull request!
- 💡 Need assistance to trace your own code? Let us know!

---

## 📞 Support

Need help? Here are your options:

- 📚 **Documentation**: You're reading it!
- 🏠 **Project Home**: Visit our main repositories for Pallas ([GitLab](http://gitlab.inria.fr/pallas/pallas) or [GitHub](https://github.com/Pallas-Trace)) and [EZTrace](https://gitlab.com/eztrace/eztrace)
- 📞 **Contact us**: [Main developer](mailto:catherine.guelque+pallas_support@telecom-sudparis.eu) 

---

*Happy tracing! 🎉*

---
