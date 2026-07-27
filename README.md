# AgroAI — AI and Automation in Agriculture

An Intelligent Framework for Sustainable Farming and Crop Management.

**Live site:** [agroaiapp.me](https://agroaiapp.me)  
**API Documentation:** [api.agroaiapp.me/docs](https://api.agroaiapp.me/docs)

---

## Overview

AgroAI is an end-to-end smart agriculture ecosystem integrating IoT soil sensing, machine learning crop recommendations, deep learning plant disease detection, microservice API gateway, and a mobile application for farmers.

### System Architecture & Modules

| Module | Description | Tech Stack |
|--------|-------------|------------|
| `web/` | Marketing homepage & Supabase web dashboard | HTML, CSS, JS, Supabase, Python |
| `Auth/` | Microservice for JWT user authentication | Python, FastAPI, PostgreSQL, JWT |
| `ApiGateway/` | Centralized gateway routing public requests | Python, FastAPI |
| `AgroSensor/` | Real-time IoT soil sensor API & dashboard | Python, PostgreSQL, Modbus |
| `Crop_Recommendation_Engine/` | Machine learning crop recommendation model | Python, Scikit-Learn, Streamlit |
| `Plant_Disease_Detection/` | Deep learning leaf disease classification | Python, PyTorch/EfficientNet, Gradio |
| `AgroMobile/` | Farmer cross-platform mobile application | Flutter, Dart |
| `oracle-cloud/` | Production cloud deployment scripts & systemd units | Shell, Systemd, Nginx |

---

## License

This project is licensed under the [MIT License](LICENSE).
