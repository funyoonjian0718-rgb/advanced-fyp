import api from "./api";

export interface LoginRequest {
    email: string;
    password: string;
}

export interface RegisterRequest {
    email: string;
    password: string;
}

export interface User {
    id: number;
    email: string;
}

export interface LoginResponse {
    access_token: string;
    token_type: string;
    user: User;
}

export async function login(
    data: LoginRequest
): Promise<LoginResponse> {
    const response = await api.post<LoginResponse>(
        "/api/auth/login",
        data
    );

    return response.data;
}

export async function register(
    data: RegisterRequest
): Promise<User> {
    const response = await api.post<User>(
        "/api/auth/register",
        data
    );

    return response.data;
}

// Logs the user out of the frontend.
//
// Our current authentication uses JWTs stored in localStorage.
// Removing the token means future protected API requests
// will no longer contain an Authorization header.
export function logout(): void {
    // Remove the JWT.
    localStorage.removeItem("access_token");

    // Remove the stored user information.
    localStorage.removeItem("user");
}