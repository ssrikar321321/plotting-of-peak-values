import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io

# Set matplotlib defaults
rcParams['font.size'] = 14
rcParams['font.family'] = 'sans-serif'
rcParams['axes.linewidth'] = 1.5

st.set_page_config(page_title="Oscilloscope Spectral Analysis", layout="wide")

st.title("Spectral Intensity & Current Waveform Analysis")
st.markdown("*Visualization of intensity peaks at different wavelengths with oscilloscope current measurements*")

# File upload section
st.sidebar.header("📁 Data Files")

# Peak intensity files
square_peak_file = st.sidebar.file_uploader("Square Peak Data", type=['csv'], key='square_peak')
triangle_peak_file = st.sidebar.file_uploader("Triangle Peak Data", type=['csv'], key='triangle_peak')

# Waveform files
square_wave_file = st.sidebar.file_uploader("Square Waveform Data", type=['csv'], key='square_wave')
triangle_wave_file = st.sidebar.file_uploader("Triangle Waveform Data", type=['csv'], key='triangle_wave')
sine_wave_file = st.sidebar.file_uploader("Sine Waveform Data", type=['csv'], key='sine_wave')

# Plot settings
st.sidebar.header("⚙️ Plot Settings")
waveforms_to_plot = st.sidebar.multiselect(
    "Select waveforms to display",
    ["Sine", "Square", "Triangle"],
    default=["Sine", "Square", "Triangle"]
)

bar_width = st.sidebar.slider("Bar width (μs)", 1.0, 8.0, 4.0, 0.5)
show_grid = st.sidebar.checkbox("Show grid", value=True)
invert_intensity = st.sidebar.checkbox("Invert intensity axis", value=True)

# Color settings
st.sidebar.subheader("🎨 Colors")
col1, col2 = st.sidebar.columns(2)
color_310 = col1.color_picker("310 nm", "#5DA5DA")
color_337 = col2.color_picker("337 nm", "#FAA43A")
color_696 = col1.color_picker("696 nm", "#60BD68")
color_current = col2.color_picker("Current", "#FF0000")

# Axis ranges
st.sidebar.subheader("📊 Axis Ranges")
time_min = st.sidebar.number_input("Time min (μs)", value=0)
time_max = st.sidebar.number_input("Time max (μs)", value=300)

# Function to load and process data
@st.cache_data
def load_peak_data(file):
    if file is not None:
        df = pd.read_csv(file)
        # Standardize column names
        df.columns = df.columns.str.strip()
        return df
    return None

@st.cache_data
def load_waveform_data(file):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        # Convert time from seconds to microseconds
        if 'Time' in df.columns:
            df['Time_us'] = df['Time'] * 1e6
        return df
    return None

# Load all data
square_peaks = load_peak_data(square_peak_file)
triangle_peaks = load_peak_data(triangle_peak_file)
square_waveform = load_waveform_data(square_wave_file)
triangle_waveform = load_waveform_data(triangle_wave_file)
sine_waveform = load_waveform_data(sine_wave_file)

# Check if we have data to plot
data_available = {
    "Sine": sine_waveform is not None,
    "Square": square_peaks is not None and square_waveform is not None,
    "Triangle": triangle_peaks is not None and triangle_waveform is not None
}

# Filter selected waveforms to only available ones
plots_to_create = [w for w in waveforms_to_plot if data_available.get(w, False)]

if len(plots_to_create) > 0:
    
    # Create figure with subplots
    num_plots = len(plots_to_create)
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot each selected waveform
    for idx, waveform_type in enumerate(plots_to_create):
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Get appropriate data
        if waveform_type == "Square":
            peak_data = square_peaks
            wave_data = square_waveform
            title_suffix = "Square Wave"
        elif waveform_type == "Triangle":
            peak_data = triangle_peaks
            wave_data = triangle_waveform
            title_suffix = "Triangle Wave"
        else:  # Sine
            peak_data = None  # Sine doesn't have peak data in uploads
            wave_data = sine_waveform
            title_suffix = "Sine Wave"
        
        # Plot intensity bars if peak data exists
        if peak_data is not None:
            time_col = [col for col in peak_data.columns if 'Time' in col][0]
            
            # Find wavelength columns
            col_310 = [col for col in peak_data.columns if '310' in col][0]
            col_337 = [col for col in peak_data.columns if '337' in col][0]
            col_696 = [col for col in peak_data.columns if '696' in col][0]
            
            time = peak_data[time_col].values
            int_310 = peak_data[col_310].values
            int_337 = peak_data[col_337].values
            int_696 = peak_data[col_696].values
            
            # Create bar plots
            width = bar_width
            ax.bar(time - width, int_310, width=width, 
                   label='310.0 nm', color=color_310, alpha=0.8, edgecolor='none')
            ax.bar(time, int_337, width=width, 
                   label='337.0 nm', color=color_337, alpha=0.8, edgecolor='none')
            ax.bar(time + width, int_696, width=width, 
                   label='696.0 nm', color=color_696, alpha=0.8, edgecolor='none')
        
        # Plot current waveform
        if wave_data is not None and 'Time_us' in wave_data.columns:
            time_wave = wave_data['Time_us'].values
            current = wave_data['Current'].values
            
            ax2.plot(time_wave, current, color=color_current, 
                    linewidth=2.5, label='Current')
        
        # Formatting
        ax.set_xlabel('Time (μs)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Current (mA)', fontsize=16, fontweight='bold', color=color_current)
        
        # Tick formatting
        ax.tick_params(axis='both', labelsize=13, width=1.5, length=6)
        ax2.tick_params(axis='y', labelcolor=color_current, labelsize=13, width=1.5, length=6)
        
        # Invert intensity axis if requested
        if invert_intensity and peak_data is not None:
            ax.invert_yaxis()
        
        # Add panel label
        panel_label = chr(97 + idx)  # a, b, c
        ax.text(0.02, 0.98, f'{panel_label})', transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        if peak_data is not None:
            ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
                     edgecolor='black', fancybox=False)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Set axis limits
        ax.set_xlim(time_min, time_max)
        
        # Title
        ax.set_title(title_suffix, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Display plot
    st.pyplot(fig)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download PNG
        buf_png = io.BytesIO()
        plt.savefig(buf_png, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_png.seek(0)
        st.download_button(
            label="📥 Download PNG (300 DPI)",
            data=buf_png,
            file_name="spectral_intensity_plot.png",
            mime="image/png"
        )
    
    with col2:
        # Download PDF
        buf_pdf = io.BytesIO()
        plt.savefig(buf_pdf, format='pdf', bbox_inches='tight')
        buf_pdf.seek(0)
        st.download_button(
            label="📥 Download PDF",
            data=buf_pdf,
            file_name="spectral_intensity_plot.pdf",
            mime="application/pdf"
        )
    
    with col3:
        # Download SVG
        buf_svg = io.BytesIO()
        plt.savefig(buf_svg, format='svg', bbox_inches='tight')
        buf_svg.seek(0)
        st.download_button(
            label="📥 Download SVG",
            data=buf_svg,
            file_name="spectral_intensity_plot.svg",
            mime="image/svg+xml"
        )
    
    # Show data tables
    st.markdown("---")
    st.subheader("📊 Data Tables")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Square Peaks", "Triangle Peaks", "Square Wave", "Triangle Wave", "Sine Wave"]
    )
    
    with tab1:
        if square_peaks is not None:
            st.dataframe(square_peaks, use_container_width=True)
        else:
            st.info("Upload square peak data file")
    
    with tab2:
        if triangle_peaks is not None:
            st.dataframe(triangle_peaks, use_container_width=True)
        else:
            st.info("Upload triangle peak data file")
    
    with tab3:
        if square_waveform is not None:
            st.dataframe(square_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(square_waveform)} rows")
        else:
            st.info("Upload square waveform data file")
    
    with tab4:
        if triangle_waveform is not None:
            st.dataframe(triangle_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(triangle_waveform)} rows")
        else:
            st.info("Upload triangle waveform data file")
    
    with tab5:
        if sine_waveform is not None:
            st.dataframe(sine_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(sine_waveform)} rows")
        else:
            st.info("Upload sine waveform data file")

else:
    st.info("👈 Please upload data files from the sidebar to generate visualizations")
    
    with st.expander("📋 Required File Formats"):
        st.markdown("""
        ### Peak Intensity Files (CSV)
        **Columns:** `Time(us), 310.0 nm, 337.0 nm, 696.00 nm`
        - Square: `square_peak_comparison.csv`
        - Triangle: `Triangle_peak_comparison.csv`
        
        ### Waveform Files (CSV)
        **Columns:** `Time, Voltage, Current`
        - Time in seconds (will be converted to μs)
        - Current in mA
        - Files:
          - `Square Power Calculation_SINGLE CHANNEL 1.csv`
          - `Triangular Power Calculation__SINGLE CHANNEL.csv`
          - `Sine Power Calculation_SINGLE CHANNEL 1.csv`
        """)

# Footer
st.markdown("---")
st.markdown("**Oscilloscope Spectral Intensity Dashboard** | Data visualization for plasma spectroscopy measurements")
