import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import AppLayout from "./layouts/AppLayout";
import ProtectedRoute from "./components/ProtectedRoute";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Chat from "./pages/Chat";
import History from "./pages/History";
import Profile from "./pages/Profile";

function App() {
    return (
        <BrowserRouter>
            <Routes>

                {/* 
                    Public routes.
                    
                    These pages can be accessed without logging in.
                */}
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />


                {/*
                    Protected routes.

                    ProtectedRoute checks whether the user
                    has an access token.

                    If there is no token:
                        → /login

                    If there is a token:
                        → allow access
                */}
                <Route element={<ProtectedRoute />}>

                    {/* 
                        AppLayout provides the common
                        sidebar/topbar structure.
                    */}
                    <Route element={<AppLayout />}>

                        <Route
                            path="/dashboard"
                            element={<Dashboard />}
                        />

                        <Route
                            path="/chat"
                            element={<Chat />}
                        />

                        <Route
                            path="/history"
                            element={<History />}
                        />

                        <Route
                            path="/profile"
                            element={<Profile />}
                        />

                    </Route>

                </Route>


                {/* 
                    If the user enters an unknown URL,
                    send them to the dashboard.
                */}
                <Route
                    path="*"
                    element={<Navigate to="/dashboard" replace />}
                />

            </Routes>
        </BrowserRouter>
    );
}

export default App;