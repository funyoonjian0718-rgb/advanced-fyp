import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { register } from "../services/auth";

function Register() {
    // Allows us to redirect the user after registration.
    const navigate = useNavigate();

    // Store the values entered in the form.
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");

    // Store an error message if registration fails.
    const [error, setError] = useState("");

    // Store a success message after registration.
    const [success, setSuccess] = useState("");

    // Used to disable the button while the API request is running.
    const [loading, setLoading] = useState(false);

    // Runs when the registration form is submitted.
    async function handleSubmit(
        event: React.FormEvent<HTMLFormElement>
    ) {
        // Stop the browser from refreshing the page.
        event.preventDefault();

        // Clear previous messages.
        setError("");
        setSuccess("");

        // Check that both passwords match before calling the backend.
        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            return;
        }

        setLoading(true);

        try {
            // Send the registration information to FastAPI.
            const user = await register({
                email,
                password,
            });

            // Registration was successful.
            setSuccess(
                `Account created successfully for ${user.email}.`
            );

            // Clear the form.
            setEmail("");
            setPassword("");
            setConfirmPassword("");

            // Wait a short moment so the user can see the
            // success message before going to the login page.
            setTimeout(() => {
                navigate("/login");
            }, 1000);

        } catch (error: any) {
            // FastAPI returns useful error information in "detail".
            setError(
                error.response?.data?.detail ||
                "Registration failed. Please try again."
            );

        } finally {
            // Stop the loading state.
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
            <div className="w-full max-w-md bg-white p-8 rounded-lg shadow">

                <h1 className="text-2xl font-bold mb-6">
                    Create Account
                </h1>

                {/* Display an error message when registration fails */}
                {error && (
                    <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
                        {error}
                    </div>
                )}

                {/* Display a success message after registration */}
                {success && (
                    <div className="mb-4 p-3 bg-green-100 text-green-700 rounded">
                        {success}
                    </div>
                )}

                <form onSubmit={handleSubmit}>

                    {/* Email */}
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

                    {/* Password */}
                    <div className="mb-4">
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
                            minLength={8}
                            required
                        />

                        <p className="text-sm text-gray-500 mt-1">
                            Password must be at least 8 characters.
                        </p>
                    </div>

                    {/* Confirm password */}
                    <div className="mb-6">
                        <label className="block mb-2 font-medium">
                            Confirm Password
                        </label>

                        <input
                            type="password"
                            value={confirmPassword}
                            onChange={(event) =>
                                setConfirmPassword(event.target.value)
                            }
                            className="w-full border rounded px-3 py-2"
                            placeholder="Confirm your password"
                            minLength={8}
                            required
                        />
                    </div>

                    {/* Submit button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        {loading ? "Creating account..." : "Register"}
                    </button>

                </form>

                {/* Link back to login */}
                <p className="mt-4 text-center">
                    Already have an account?{" "}
                    <button
                        type="button"
                        onClick={() => navigate("/login")}
                        className="text-blue-600 hover:underline"
                    >
                        Login
                    </button>
                </p>

            </div>
        </div>
    );
}

export default Register;