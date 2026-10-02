import { useEffect, useState } from "react";

import api from "../services/api";

interface User {
    id: number;
    email: string;
}

function Profile() {
    // Store the user returned by the backend.
    const [user, setUser] = useState<User | null>(null);

    // Store an error if the request fails.
    const [error, setError] = useState("");

    useEffect(() => {
        // Load the currently authenticated user
        // when this page is opened.
        async function loadUser() {
            try {
                // We do NOT manually add the JWT here.
                //
                // The Axios interceptor in api.ts
                // automatically adds it for us.
                const response = await api.get<User>(
                    "/api/users/me"
                );

                setUser(response.data);

            } catch (error: any) {
                setError(
                    error.response?.data?.detail ||
                    "Failed to load user."
                );
            }
        }

        loadUser();
    }, []);

    return (
        <div className="p-6">

            <h1 className="text-2xl font-bold mb-6">
                Profile
            </h1>

            {error && (
                <p className="text-red-600">
                    {error}
                </p>
            )}

            {user && (
                <div className="bg-white p-6 rounded-lg shadow">
                    <p>
                        <strong>User ID:</strong> {user.id}
                    </p>

                    <p>
                        <strong>Email:</strong> {user.email}
                    </p>
                </div>
            )}

        </div>
    );
}

export default Profile;