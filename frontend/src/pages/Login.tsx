import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { login } from "../services/auth";

function Login() {
    // Used to redirect the user to another page after login.
    const navigate = useNavigate();

    // Store the email entered by the user.
    const [email, setEmail] = useState("");

    // Store the password entered by the user.
    const [password, setPassword] = useState("");

    // Store an error message if login fails.
    const [error, setError] = useState("");

    // Store whether the login request is currently being sent.
    const [loading, setLoading] = useState(false);

    // This function runs when the user submits the login form.
    async function handleSubmit(
        event: React.FormEvent<HTMLFormElement>
    ) {
        // Prevent the browser from refreshing the page.
        event.preventDefault();

        // Clear any previous error message.
        setError("");

        // Show the loading state.
        setLoading(true);

        try {
            // Send the email and password to our FastAPI login endpoint.
            const response = await login({
                email,
                password,
            });

            // The backend returns a JWT access token.
            // We temporarily store it in localStorage so the
            // frontend can use it for authenticated API requests.
            localStorage.setItem(
                "access_token",
                response.access_token
            );

            // Store the user's basic information as well.
            localStorage.setItem(
                "user",
                JSON.stringify(response.user)
            );

            // Login was successful, so go to the dashboard.
            navigate("/dashboard");

        } catch (error: any) {
            // If FastAPI returns an error, display it to the user.
            setError(
                error.response?.data?.detail ||
                "Login failed. Please check your email and password."
            );

        } finally {
            // Stop the loading state whether login succeeded or failed.
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
            <div className="w-full max-w-md bg-white p-8 rounded-lg shadow">

                <h1 className="text-2xl font-bold mb-6">
                    Login
                </h1>

                {/* Display an error message if login fails */}
                {error && (
                    <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit}>

                    {/* Email input */}
                    <div className="mb-4">
                        <label className="block mb-2 font-medium">
                            Email
                        </label>

                        <input
                            type="email"
                            value={email}
                            onChange={(event) =>
                                setEmail(event.target.value)
                            }
                            className="w-full border rounded px-3 py-2"
                            placeholder="Enter your email"
                            required
                        />
                    </div>

                    {/* Password input */}
                    <div className="mb-6">
                        <label className="block mb-2 font-medium">
                            Password
                        </label>

                        <input
                            type="password"
                            value={password}
                            onChange={(event) =>
                                setPassword(event.target.value)
                            }
                            className="w-full border rounded px-3 py-2"
                            placeholder="Enter your password"
                            required
                        />
                    </div>

                    {/* Login button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        {loading ? "Logging in..." : "Login"}
                    </button>

                </form>

                {/* Link to registration page */}
                <p className="mt-4 text-center">
                    Don't have an account?{" "}
                    <button
                        type="button"
                        onClick={() => navigate("/register")}
                        className="text-blue-600 hover:underline"
                    >
                        Register
                    </button>
                </p>

            </div>
        </div>
    );
}

export default Login;