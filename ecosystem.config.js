module.exports = {
  apps: [
    {
      name: 'rag-chatbot-backend',
      script: 'backend/main.py',
      interpreter: 'python',
      cwd: 'C:/Apache24/htdocs/rag-chatbot',
      watch: true,
      env: {
        PYTHONPATH: '.',
        NODE_ENV: 'production'
      },
      error_file: 'logs/pm2-error.log',
      out_file: 'logs/pm2-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss'
    }
  ]
};
