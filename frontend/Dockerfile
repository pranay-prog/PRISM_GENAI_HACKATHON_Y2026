FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY . .
# The browser talks to the backend directly, so this is the host-visible URL.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
