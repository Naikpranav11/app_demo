
# 📊 Fraud Detection Admin Dashboard

This is a Flask-based web application that helps administrators manage customers, create payment links, monitor fraudulent activities, and visualize key metrics using charts.

---

## 🚀 Features
- Admin login system
- Customer management
- Fraud detection overview
- Dashboard analytics with charts (dummy data for now)
- SQLite-based storage

---

## 📁 Project Structure
```
app_demo/
│
├── app.py                 # Main Flask application
├── templates/
│   └── admin_dashboard.html
├── static/
│   └── (optional static files like images or custom CSS/JS)
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

1. **Clone the repository**:
   ```bash
   https://github.com/Naikpranav11/app_demo.git
   cd app_demo
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the dashboard**:
   Open your browser and visit:  
   `http://127.0.0.1:5000/admin/dashboard`

---

## 🛠 Notes
- Uses SQLite database (created automatically).
- Replace dummy chart data with real analytics as needed.
- You can expand it with user authentication, model predictions, etc.
