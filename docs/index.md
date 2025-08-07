
# Pallas Documentation 🚀

Welcome to the documentation for **Pallas**, a trace format tailored for Exascale tracing.

---

## What is Pallas?

Pallas is a state-of-the-art tracing format designed to help developers and researchers analyze the performance of parallel applications. Whether you're working with MPI, OpenMP, CUDA, or other parallel paradigms, Pallas is able to scalably trace and analyze your application's behavior.
Pallas is a library offering a complete OTF2 interface, so that it can be used with any tracing tool relying on the OTF2 format. In this guide, we rely on [EZTrace](https://gitlab.com/eztrace/eztrace) to generate either OTF2 or Pallas traces.


---

## 📚 Documentation Structure

This documentation is organized into six main sections, presenting both how to install EZTrace, Pallas, OTF2 and how to trace in these formats and perform trace analysis.

### 🏃‍♂️ [Quick Start](01-quick-start.md)
Quickly get up and running with EZTrace+Pallas.

### 📊 [OTF2 Tracing with EZTrace](02-tracing-otf2/index.md)
Learn how to generate traces in the standard **Open Trace Format 2 (OTF2)** using the EZTrace tracing tool. This section covers:
- 🔧 [Installing OTF2](02-tracing-otf2/01-installing-otf2/index.md)
- ⚙️ [Installing EZTrace](02-tracing-otf2/02-installing-eztrace/index.md)
- 🌐 [Tracing MPI Applications](02-tracing-otf2/03-tracing-mpi/index.md)
- ⚡ [Tracing Other Parallel Applications](02-tracing-otf2/04-tracing-other-parallel/index.md)
- 🔌 [Create Your Own EZTrace Plugin](02-tracing-otf2/05-create-plugin/index.md)

### 📈 [Pallas Tracing with EZTrace](03-tracing-pallas/index.md)
Discover how to trance in the  **Pallas trace format** for low-overhead and compact traces:
- 🎭 [Pallas Format Overview](03-tracing-pallas/01-presentation/index.md)
- 🔧 [Installing Pallas](03-tracing-pallas/02-installing-pallas/index.md)
- ⚙️ [Installing EZTrace + Pallas](03-tracing-pallas/03-installing-eztrace-pallas/index.md)
- 🌐 [MPI Application Tracing](03-tracing-pallas/04-tracing-mpi/index.md)
- ⚡ [Other Parallel Apps](03-tracing-pallas/05-tracing-other-parallel/index.md)
- 📚 [Custom Library Tracing](03-tracing-pallas/06-trace-own-library/index.md)

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

## 🚀 Getting Started

New to Pallas? Start here:

1. **🏃‍♂️ **Quick Start Guide**: - [Get running in 5 minutes](01-quick-start.md)
2. **🔧 Choose your format and trace with EZTrace**: [OTF2](02-tracing-otf2/index.md) or [Pallas](03-tracing-pallas/index.md)
3. **🔍 Scalable trace analysis** with [Pallas native tools](04-analyzing-pallas/index.md)
4. **📊 Trace visualization** at scale with [Blup](05-visualizing-blup/index.md)

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
- 🏠 **Project Home**: Visit our main repositories for [Pallas](https://github.com/Pallas-Trace) and [EZTrace](https://gitlab.com/eztrace/eztrace)

---

*Happy tracing! 🎉*

---



<sub><sup>Theme by [Swissquote](https://github.com/swissquote/swissquote-daux-theme)</sup></sub>
