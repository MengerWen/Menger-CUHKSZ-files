import { defineConfig } from "vite";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const defaultBase =
  process.env.SHOWCASE_BASE_PATH ??
  (process.env.GITHUB_ACTIONS && repositoryName ? `/${repositoryName}/` : "/");

export default defineConfig({
  base: defaultBase,
  plugins: [
    {
      name: "pre-presentation-route",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url ?? "";
          if (url === "/pre" || url === "/pre/" || url.startsWith("/pre?") || url.startsWith("/pre/?")) {
            const queryIndex = url.indexOf("?");
            const query = queryIndex >= 0 ? url.slice(queryIndex) : "";
            res.statusCode = 302;
            res.setHeader("Location", `/pre/index.html${query}`);
            res.end();
            return;
          }
          next();
        });
      },
    },
  ],
  server: {
    port: 5173,
  },
});
