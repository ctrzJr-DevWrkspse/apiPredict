# Stage 1: Build
FROM node:20 AS build
WORKDIR /app

# Copy package files and install ALL deps (including dev)
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build frontend
RUN npm run build


# Stage 2: Production
FROM node:20-slim AS prod
WORKDIR /app

# Copy only built frontend
COPY --from=build /app/dist ./dist

# If you also have a backend (e.g., gunicorn for Python API),
# you’ll need a separate stage for that.
# Otherwise, serve frontend with something like serve:
RUN npm install -g serve

CMD ["serve", "-s", "dist", "-l", "3000"]
