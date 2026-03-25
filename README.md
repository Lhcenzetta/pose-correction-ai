# PoseCorrect: AI-Powered Pose Correction

`PoseCorrect` is a professional, AI-driven application designed to help users perform physical therapy and fitness exercises with perfect form. By leveraging real-time pose estimation and deep learning classification, the app provides instant, clinical-grade feedback to ensure safety and maximize the effectiveness of every movement.

---

## 🚀 Key Features

- **Real-time Pose Tracking**: High-fidelity joint tracking using MediaPipe.
- **AI Form Analysis**: Custom TensorFlow models classify movements and detect deviations from proper technique.
- **Instant Clinical Feedback**: Dynamic verbal and visual cues (e.g., "Raise LEFT arm more") based on precise joint angle calculations.
- **Session Management**: Full lifecycle support for exercise sessions—start, track, and finalize with detailed performance summaries.
- **ML Performance Tracking**: Integrated with **MLflow** to monitor model inference and user progress over time.
- **Secure Authentication**: Robust user login and signup system using JWT.
- **Premium Medical UI**: A clean, responsive, and intuitive interface built for healthcare and fitness contexts.

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: [Next.js 15+](https://nextjs.org/) (App Router)
- **Library**: [React 19](https://react.dev/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **ORM**: [SQLAlchemy](https://www.sqlalchemy.org/)
- **Database**: [PostgreSQL](https://www.postgresql.org/)
- **Auth**: JWT (JSON Web Tokens)

### AI & Machine Learning
- **Pose Estimation**: [MediaPipe](https://google.github.io/mediapipe/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/)
- **Experiment Tracking**: [MLflow](https://mlflow.org/)

### DevOps & Infrastructure
- **Containerization**: [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- **CI/CD**: [GitHub Actions](https://github.com/features/actions)

---

## 📂 Project Structure

```text
pose-correction-ai/
├── backend/                # FastAPI application
│   ├── DL/                 # Deep Learning models & pipeline
│   ├── db/                 # Database configuration & session
│   ├── models/             # SQLAlchemy database models
│   ├── routers/            # API endpoints (Auth, Sessions, Exercises)
│   └── schema/             # Pydantic validation schemas
├── frontend/               # Next.js application
│   ├── src/app/            # App Router pages
│   └── src/components/     # Shared React components
├── tests/                  # Backend test suite
├── docker-compose.yml      # Orchestration for DB, Backend, Frontend, MLflow
└── init.sql                # Initial database schema and data
```

---

## ⚙️ Getting Started

### Prerequisites
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.9+ (for local development)
- Node.js 18+ (for local development)

### Environment Setup
Create a `.env` file in the root directory and configure the following variables (refer to `docker-compose.yml`):
```env
user=your_db_user
password=your_db_password
database=pose_correct_db
SECRET_KEY=your_jwt_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### Running with Docker
The easiest way to start the entire stack (PostgreSQL, Backend, Frontend, and MLflow) is using Docker Compose:

```bash
docker-compose up --build
```

- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000/docs`
- **MLflow Dashboard**: `http://localhost:5000`

---

## 🧪 Testing

The project includes a comprehensive test suite for the backend.

```bash
# Run tests using pytest (requires local venv with requirements.txt installed)
export TESTING=true
pytest
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
