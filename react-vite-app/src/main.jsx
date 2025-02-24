import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './main.css'
import App from './home.jsx'
import Header from './header.jsx'
import Footer from './footer.jsx'

/*This file is to combine all of the files together*/
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Header />  
    <App />
    <Footer />
  </StrictMode>,
)
