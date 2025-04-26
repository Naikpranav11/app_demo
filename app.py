from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ======= Database Models =======

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first = db.Column(db.String(100))
    last = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    street = db.Column(db.String(200))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    zip = db.Column(db.String(20))
    lat = db.Column(db.Float)
    long = db.Column(db.Float)
    unix_time = db.Column(db.Float)
    trans_hour = db.Column(db.Integer)
    trans_day = db.Column(db.Integer)
    age = db.Column(db.Integer)
    amt = db.Column(db.Float)
    is_fraud = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# ====== Load Model and Preprocessors ======

try:
    model = load_model('models/credit_fraud_cnn_model_resampled.h5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    with open('models/scaler_resampled.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit(1)

try:
    with open('models/label_encoders_resampled.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    print(f"Error loading label encoders: {e}")
    exit(1)

feature_columns = [
    'cc_num', 'amt', 'first', 'last', 'gender', 'street', 'city', 'state',
    'zip', 'lat', 'long', 'unix_time', 'trans_hour', 'trans_day', 'age'
]

# ========== Customer Routes ==========

@app.route('/')
def index():
    return redirect('/admin/login')

@app.route('/submit_transaction', methods=['POST'])
def predict():
    try:
        input_data = {
            'cc_num': float(request.form['cc_num']),
            'amt': float(request.form['amt']),
            'first': request.form['first'],
            'last': request.form['last'],
            'gender': request.form['gender'],
            'street': request.form['street'],
            'city': request.form['city'],
            'state': request.form['state'],
            'zip': float(request.form['zip']),
            'lat': float(request.form['lat']),
            'long': float(request.form['long']),
            'unix_time': float(request.form['unix_time']),
            'trans_hour': float(request.form['trans_hour']),
            'trans_day': float(request.form['trans_day']),
            'age': float(request.form['age'])
        }

        df_input = pd.DataFrame([input_data])

        # Encode categoricals
        categorical_columns = ['first', 'last', 'gender', 'street', 'city', 'state']
        for column in categorical_columns:
            if column in label_encoders:
                try:
                    df_input[column] = label_encoders[column].transform([df_input[column].iloc[0]])[0]
                except ValueError:
                    df_input[column] = 0

        X = df_input[feature_columns].values
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape((1, X_scaled.shape[1], 1))

        prediction = model.predict(X_reshaped)
        is_fraud = int(prediction[0][0] > 0.0001)

        # Save to database
        new_customer = Customer(
            first=input_data['first'],
            last=input_data['last'],
            gender=input_data['gender'],
            street=input_data['street'],
            city=input_data['city'],
            state=input_data['state'],
            zip=str(int(input_data['zip'])),
            lat=input_data['lat'],
            long=input_data['long'],
            unix_time=input_data['unix_time'],
            trans_hour=int(input_data['trans_hour']),
            trans_day=int(input_data['trans_day']),
            age=int(input_data['age']),
            amt=input_data['amt'],
            is_fraud=bool(is_fraud)
        )
        db.session.add(new_customer)
        db.session.commit()

        # Decide Status
        if is_fraud:
            status = 'Rejected'
        else:
            status = 'Accepted'

        return render_template('result.html', status=status)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ========== Admin Routes ==========

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        existing_admin = Admin.query.filter_by(email=email).first()
        if existing_admin:
            flash('Email already registered.', 'danger')
            return redirect(url_for('admin_register'))

        new_admin = Admin(email=email, password=password)
        db.session.add(new_admin)
        db.session.commit()

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('admin_login'))

    return render_template('admin_register.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        admin = Admin.query.filter_by(email=email, password=password).first()
        if not admin:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('admin_login'))

        session['admin'] = admin.email
        flash('Logged in successfully.', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    # Get the total number of customers
    total_customers = Customer.query.count()
    
    # Get the total number of fraud customers
    total_frauds = Customer.query.filter_by(is_fraud=True).count()
    
    # Get the total payments (sum of the 'amt' column)
    total_payments = db.session.query(db.func.sum(Customer.amt)).scalar() or 0
    
    # Calculate successful transactions (total customers - fraud customers)
    successful_transactions = total_customers - total_frauds
    
    # Calculate the fraud percentage
    fraud_percentage = (total_frauds / total_customers * 100) if total_customers > 0 else 0
    
    # Get payments over time (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_transactions = Customer.query.filter(Customer.timestamp >= thirty_days_ago).all()
    
    # Prepare data for payments over time (group by date)
    payments_by_date = {}
    for transaction in recent_transactions:
        date = transaction.timestamp.strftime('%Y-%m-%d')
        if date not in payments_by_date:
            payments_by_date[date] = 0
        payments_by_date[date] += transaction.amt
    
    # Extract the dates and payment amounts for the chart
    dates = list(payments_by_date.keys())
    payments = list(payments_by_date.values())

    # Render the template with the data
    return render_template(
        'admin_dashboard.html',
        total_customers=total_customers,
        total_frauds=total_frauds,
        total_payments=round(total_payments, 2),
        successful_transactions=successful_transactions,
        fraud_percentage=round(fraud_percentage, 2),
        dates=dates,
        payments=payments
    )

@app.route('/admin/customers')
def admin_customers():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    customers = Customer.query.filter_by(is_fraud=False).all()
    return render_template('customer_list.html', customers=customers)

@app.route('/admin/fraud')
def admin_fraud():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    customers = Customer.query.filter_by(is_fraud=True).all()
    return render_template('fraud_customers.html', customers=customers)

@app.route('/admin/create_link')
def create_link():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    link = request.host_url + 'customer_form'
    return render_template('create_link.html', link=link)

@app.route('/customer_form', methods=['GET', 'POST'])
def customer_form():
    if request.method == 'POST':
        return redirect('/success')
    return render_template('customer_form.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('admin_login'))

# ========== Main Runner ==========

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)





# from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
# import os
# import numpy as np
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'

# # ========== Load Model and Preprocessing Objects ==========

# try:
#     model = load_model('models\credit_fraud_cnn_model.h5', compile=False)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# try:
#     with open('models\scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
# except Exception as e:
#     print(f"Error loading scaler: {e}")
#     exit(1)

# try:
#     with open('models\label_encoders.pkl', 'rb') as f:
#         label_encoders = pickle.load(f)
# except Exception as e:
#     print(f"Error loading label encoders: {e}")
#     exit(1)

# # ========== In-memory Databases ==========

# admin_db = {}
# customer_db = []
# fraudulent_customers = []

# # ========== Feature Columns ==========

# feature_columns = [
#     'cc_num', 'amt', 'first', 'last', 'gender', 'street', 'city', 'state',
#     'zip', 'lat', 'long', 'unix_time', 'trans_hour', 'trans_day', 'age'
# ]

# # ========== Customer Routes ==========

# @app.route('/')
# def index():
#     return redirect('/admin/login')

# @app.route('/submit_transaction', methods=['POST'])
# def predict():
#     try:
#         input_data = {
#             'cc_num': float(request.form['cc_num']),
#             'amt': float(request.form['amt']),
#             'first': request.form['first'],
#             'last': request.form['last'],
#             'gender': request.form['gender'],
#             'street': request.form['street'],
#             'city': request.form['city'],
#             'state': request.form['state'],
#             'zip': float(request.form['zip']),
#             'lat': float(request.form['lat']),
#             'long': float(request.form['long']),
#             'unix_time': float(request.form['unix_time']),
#             'trans_hour': float(request.form['trans_hour']),
#             'trans_day': float(request.form['trans_day']),
#             'age': float(request.form['age'])
#         }

#         df_input = pd.DataFrame([input_data])

#         # Encode categorical
#         categorical_columns = ['first', 'last', 'gender', 'street', 'city', 'state']
#         for column in categorical_columns:
#             if column in label_encoders:
#                 try:
#                     df_input[column] = label_encoders[column].transform([df_input[column].iloc[0]])[0]
#                 except ValueError:
#                     df_input[column] = 0

#         X = df_input[feature_columns].values
#         X_scaled = scaler.transform(X)
#         X_reshaped = X_scaled.reshape((1, X_scaled.shape[1], 1))

#         prediction = model.predict(X_reshaped)
#         is_fraud = int(prediction[0][0] > 0.5)

#         if is_fraud:
#             fraud_count = 1
#         else:
#             fraud_count = 0

#         customer_record = input_data.copy()
#         customer_record.pop('cc_num')  # Do not save credit card number
#         customer_record['is_fraud'] = bool(is_fraud)

#         if fraud_count > 2:
#             status = 'Rejected'
#             fraudulent_customers.append(customer_record)
#         elif fraud_count <= 2 and fraud_count > 0:
#             status = 'Pending Admin Review'
#             customer_db.append(customer_record)
#         else:
#             status = 'Accepted'
#             customer_db.append(customer_record)

#         return render_template('result.html', status=status)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # ========== Admin Routes (Registration, Login, Dashboard) ==========

# @app.route('/admin/register', methods=['GET', 'POST'])
# def admin_register():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         if email in admin_db:
#             flash('Email already registered.', 'danger')
#             return redirect(url_for('admin_register'))

#         # Save admin (no OTP for now)
#         admin_db[email] = password
#         flash('Registration successful. Please login.', 'success')
#         return redirect(url_for('admin_login'))

#     return render_template('admin_register.html')

# @app.route('/admin/login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         if email not in admin_db or admin_db[email] != password:
#             flash('Invalid credentials.', 'danger')
#             return redirect(url_for('admin_login'))

#         session['admin'] = email
#         flash('Logged in successfully.', 'success')
#         return redirect(url_for('admin_dashboard'))

#     return render_template('admin_login.html')

# @app.route('/admin/dashboard')
# def admin_dashboard():
#     if 'admin' not in session:
#         return redirect(url_for('admin_login'))
#     return render_template('admin_dashboard.html')

# @app.route('/admin/customers')
# def admin_customers():
#     if 'admin' not in session:
#         return redirect(url_for('admin_login'))
#     return render_template('customer_list.html', customers=customer_db)

# @app.route('/admin/fraud')
# def admin_fraud():
#     if 'admin' not in session:
#         return redirect(url_for('admin_login'))
#     return render_template('fraud_customers.html', customers=fraudulent_customers)

# @app.route('/admin/create_link', methods=['GET', 'POST'])
# def create_link():
#     if request.method == 'POST':
#         link = request.host_url + 'customer_form'  # dynamically generates based on where app is running
#         return render_template('create_link.html', link=link)
#     return render_template('create_link.html', link=None)

# @app.route('/customer_form', methods=['GET', 'POST'])
# def customer_form():
#     if request.method == 'POST':
#         # handle the form submission here
#         return redirect('/success')  # or wherever you want to go after form submission
#     return render_template('customer_form.html')

# @app.route('/admin/logout')
# def admin_logout():
#     session.pop('admin', None)
#     flash('Logged out successfully.', 'success')
#     return redirect(url_for('admin_login'))

# # ========== Main Runner ==========

# if __name__ == '__main__':
#     app.run(debug=True)

# # # app.py
# # from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# # from flask_sqlalchemy import SQLAlchemy
# # from flask_mail import Mail, Message
# # import numpy as np
# # import pandas as pd
# # import pickle
# # import os
# # import random
# # import string
# # from utils import load_model_and_preprocessors, preprocess_input, send_otp

# # # Initialize Flask app
# # app = Flask(__name__)
# # app.secret_key = 'your_secret_key_here'
# # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # # Flask-Mail config
# # app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# # app.config['MAIL_PORT'] = 587
# # app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
# # app.config['MAIL_PASSWORD'] = 'your_app_password'
# # app.config['MAIL_USE_TLS'] = True
# # app.config['MAIL_USE_SSL'] = False

# # # Initialize extensions
# # db = SQLAlchemy(app)
# # mail = Mail(app)

# # # Load model and preprocessing
# # model, scaler, label_encoders = load_model_and_preprocessors()

# # # Database models
# # class Admin(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)
# #     email = db.Column(db.String(120), unique=True, nullable=False)
# #     password = db.Column(db.String(80), nullable=False)

# # class Customer(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)
# #     name = db.Column(db.String(200))
# #     gender = db.Column(db.String(10))
# #     street = db.Column(db.String(200))
# #     city = db.Column(db.String(100))
# #     state = db.Column(db.String(100))
# #     zip_code = db.Column(db.String(20))
# #     status = db.Column(db.String(20))  # Accepted, Rejected, Review

# # # -------------- Routes ---------------- #

# # @app.route('/')
# # def home():
# #     return render_template('customer_form.html')

# # @app.route('/submit_transaction', methods=['POST'])
# # def submit_transaction():
# #     try:
# #         data = request.form.to_dict()
# #         session['customer_data'] = data
# #         return redirect(url_for('processing'))
# #     except Exception as e:
# #         flash(str(e), 'danger')
# #         return redirect(url_for('home'))

# # @app.route('/processing')
# # def processing():
# #     return render_template('processing.html')

# # @app.route('/analyze_transaction')
# # def analyze_transaction():
# #     try:
# #         customer_data = session.get('customer_data', None)
# #         if not customer_data:
# #             return redirect(url_for('home'))

# #         processed_input = preprocess_input(customer_data, label_encoders, scaler)
# #         prediction = model.predict(processed_input)
# #         fraud_count = int(prediction[0][0] > 0.5)

# #         # Decision based on fraud count
# #         if fraud_count > 2:
# #             status = "Rejected"
# #         elif fraud_count <= 2 and fraud_count > 0:
# #             status = "Review"
# #         else:
# #             status = "Accepted"

# #         # Save customer info without cc_num
# #         customer = Customer(
# #             name=customer_data['first'] + ' ' + customer_data['last'],
# #             gender=customer_data['gender'],
# #             street=customer_data['street'],
# #             city=customer_data['city'],
# #             state=customer_data['state'],
# #             zip_code=customer_data['zip'],
# #             status=status
# #         )
# #         db.session.add(customer)
# #         db.session.commit()

# #         return render_template('result.html', status=status)
# #     except Exception as e:
# #         flash(str(e), 'danger')
# #         return redirect(url_for('home'))

# # # -------------- Admin Routes ------------- #

# # @app.route('/admin/register', methods=['GET', 'POST'])
# # def admin_register():
# #     if request.method == 'POST':
# #         email = request.form['email']
# #         password = request.form['password']

# #         if email in admin_db:
# #             flash('Email already registered.', 'danger')
# #             return redirect(url_for('admin_register'))

# #         # Store email and password directly (no OTP for now)
# #         admin_db[email] = password
# #         flash('Registration successful. Please login.', 'success')
# #         return redirect(url_for('admin_login'))

# #     return render_template('admin_register.html')

# # # @app.route('/admin/register', methods=['GET', 'POST'])
# # # def admin_register():
# # #     if request.method == 'POST':
# # #         email = request.form['email']
# # #         password = request.form['password']
# # #         otp = send_otp(email, mail)
# # #         session['register_otp'] = otp
# # #         session['register_email'] = email
# # #         session['register_password'] = password
# # #         return render_template('admin_verify.html')
# # #     return render_template('admin_register.html')

# # @app.route('/admin/verify', methods=['POST'])
# # def admin_verify():
# #     entered_otp = request.form['otp']
# #     if entered_otp == session.get('register_otp'):
# #         new_admin = Admin(email=session['register_email'], password=session['register_password'])
# #         db.session.add(new_admin)
# #         db.session.commit()
# #         flash('Registration successful!', 'success')
# #         return redirect(url_for('admin_login'))
# #     else:
# #         flash('Invalid OTP. Try again.', 'danger')
# #         return render_template('admin_verify.html')

# # @app.route('/admin/login', methods=['GET', 'POST'])
# # def admin_login():
# #     if request.method == 'POST':
# #         email = request.form['email']
# #         password = request.form['password']

# #         if email not in admin_db or admin_db[email] != password:
# #             flash('Invalid credentials.', 'danger')
# #             return redirect(url_for('admin_login'))

# #         session['admin'] = email
# #         flash('Logged in successfully.', 'success')
# #         return redirect(url_for('admin_dashboard'))

# #     return render_template('admin_login.html')

# # # @app.route('/admin/login', methods=['GET', 'POST'])
# # # def admin_login():
# # #     if request.method == 'POST':
# # #         email = request.form['email']
# # #         password = request.form['password']
# # #         admin = Admin.query.filter_by(email=email, password=password).first()
# # #         if admin:
# # #             otp = send_otp(email, mail)
# # #             session['login_otp'] = otp
# # #             session['admin_email'] = email
# # #             return render_template('admin_login_verify.html')
# # #         else:
# # #             flash('Invalid login credentials.', 'danger')
# # #             return redirect(url_for('admin_login'))
# # #     return render_template('admin_login.html')

# # @app.route('/admin/login/verify', methods=['POST'])
# # def admin_login_verify():
# #     entered_otp = request.form['otp']
# #     if entered_otp == session.get('login_otp'):
# #         session['admin_logged_in'] = True
# #         return redirect(url_for('admin_dashboard'))
# #     else:
# #         flash('Invalid OTP.', 'danger')
# #         return render_template('admin_login_verify.html')

# # @app.route('/admin/dashboard')
# # def admin_dashboard():
# #     if not session.get('admin_logged_in'):
# #         return redirect(url_for('admin_login'))
# #     return render_template('admin_dashboard.html')

# # @app.route('/admin/customers')
# # def admin_customers():
# #     if not session.get('admin_logged_in'):
# #         return redirect(url_for('admin_login'))
# #     customers = Customer.query.filter(Customer.status != 'Rejected').all()
# #     return render_template('customer_list.html', customers=customers)

# # @app.route('/admin/fraud_customers')
# # def admin_fraud_customers():
# #     if not session.get('admin_logged_in'):
# #         return redirect(url_for('admin_login'))
# #     customers = Customer.query.filter(Customer.status == 'Rejected').all()
# #     return render_template('fraud_customers.html', customers=customers)

# # @app.route('/admin/create_link')
# # def admin_create_link():
# #     if not session.get('admin_logged_in'):
# #         return redirect(url_for('admin_login'))
# #     payment_link = request.url_root
# #     return render_template('create_link.html', payment_link=payment_link)

# # # ---------------- Run ------------------ #

# # if __name__ == "__main__":
# #     if not os.path.exists('database.db'):
# #         db.create_all()
# #     app.run(debug=True)
