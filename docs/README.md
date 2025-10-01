
# Pallas Documentation 🚀

Welcome to the documentation for **Pallas**, a trace format tailored for HPC tracing.

---

## What is Pallas?

Pallas is a state-of-the-art tracing format designed to help developers
and researchers analyze the performance of parallel applications.
Whether you're working with MPI, OpenMP, CUDA, or other parallel paradigms,
Pallas is able to trace and analyze your application's behavior, even at larger scales.
Pallas is a library offering a OTF2 interface, so that it can be used with any tracing tool relying on the OTF2 format.
In this guide, we rely on [EZTrace](https://gitlab.com/eztrace/eztrace) to generate either OTF2 or Pallas traces.
However, Pallas can be used without EZTrace, and we will not give a guide to EZTrace either.

---

## 🚀 Getting Started

New to Pallas? Start here:

1. **🏃‍♂️ Quick Start Guide**: - [Get running in 5 minutes](01-quick-start.md)
2. **🔧 Choose your format and trace with EZTrace**: [Using EZTrace/Pallas](02-pallas/index.md)
3. **🔍 Scalable trace analysis** with [Pallas native tools](04-analyzing-pallas/index.md)
4. **📊 Trace visualization** at scale with [Blup](05-visualizing-blup/index.md)

---

## 📚 Table of Contents

### 🏃‍♂️ [Quick Start](01-quick-start.md)
Quickly get up and running with Pallas and EZTrace


### 📈 [Pallas Tracing with EZTrace](02-pallas/)
Learn how to use the **Pallas trace format** to speed up you analyses:
- 🎭 [Pallas Format Overview](02-pallas/01-presentation.md): All you need to understand Pallas
- 🔧 [Installing Pallas](02-pallas/02-installing-pallas/index.md)
- ⚙️ [Installing EZTrace + Pallas]() (TODO)
- Tracing your applications by yourself:
    - ⚡ [Multithreaded Application Tracing (OpenMP)](02-pallas/03-tracing-examples/01-multithread/index.md)
    - 🌐 [Multiprocess Application Tracing (MPI)](02-pallas/03-tracing-examples/02-multiprocess/index.md)
    - 📚 [Custom Library Tracing](02-pallas/03-tracing-examples/03-custom-library/index.md)

### 🔍 [Scalable Trace Analysis with Pallas Traces](04-analyzing-pallas/index.md)
Perform scalable trace analysis using the **Pallas trace format** :
- 🎯 [Analysis Fundamentals](04-analyzing-pallas/01-generalities/index.md)
- 🛠️ [Native Analysis Tools](04-analyzing-pallas/02-native-analysis/index.md)
- 🏗️ [Custom Analysis Programs](04-analyzing-pallas/03-build-analysis/index.md)

### 📊 [Scalable Visualization of Pallas Traces with Blup](05-visualizing-blup/index.md)
Scalable trace visualization relying on the **Pallas API** using the **Blup** tool:
- 🎨 [Blup Overview](05-visualizing-blup/01-generalities/index.md)
- 📂 [Opening Traces](05-visualizing-blup/02-open-trace/index.md)
- ✨ [Advanced Features](05-visualizing-blup/03-other-functionalities/index.md)

### 📖 [API Reference](06-api-reference/index.md)
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

- 📝 Found a bug?Want to improve docs? Submit a pull request!
- 💡 Need assistance to trace your own code? Let us know!

---

## 📞 Support

Need help? Here are your options:

- 📚 **Documentation**: You're reading it!
- 🏠 **Project Home**: Visit our main repositories for Pallas ([GitLab](http://gitlab.inria.fr/pallas/pallas) or [GitHub](https://github.com/Pallas-Trace)) and [EZTrace](https://gitlab.com/eztrace/eztrace)
- 📞 **Contact us**: [Main developer](mailto://catherine.guelque+pallas_support@telecom-sudparis.eu) 

---

*Happy tracing! 🎉*

---
