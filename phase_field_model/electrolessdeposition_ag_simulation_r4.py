# ----------------------------------------------------------------------
#  POST-PROCESSOR : material field + electric potential (proxy)
# ----------------------------------------------------------------------
if st.session_state.snapshots:
    # ------------------------------------------------------------------
    #  Helper – combine phi/psi into a single material index
    # ------------------------------------------------------------------
    def build_material(phi, psi, method):
        """
        phi : Ag-shell  (0 … 1)
        psi : Cu-core   (0 … 1)
        method : string → one of the formulas below
        Returns a 2-D (or 3-D) array with values
            0 → electrolyte
            1 → Ag-shell
            2 → Cu-core
        """
        if method == "phi + 2*psi (simple)":
            return phi + 2.0 * psi          # 0 / 1 / 2
        elif method == "phi*(1-psi) + 2*psi":
            return phi * (1.0 - psi) + 2.0 * psi
        elif method == "h·(phi² + psi²)":
            h = 0.5                     # you can expose this as a slider later
            return h * (phi**2 + psi**2)
        elif method == "max(phi, psi) + psi":
            # guarantees core = 2, shell = 1, electrolyte = 0
            return np.where(psi > 0.5, 2.0,
                   np.where(phi > 0.5, 1.0, 0.0))
        else:
            raise ValueError("unknown material method")

    # ------------------------------------------------------------------
    #  UI – material composition + potential
    # ------------------------------------------------------------------
    st.subheader("Material composition & electric-potential proxy")

    col_a, col_b = st.columns([2, 2])
    with col_a:
        material_method = st.selectbox(
            "Material interpolation",
            ["phi + 2*psi (simple)",
             "phi*(1-psi) + 2*psi",
             "h·(phi² + psi²)",
             "max(phi, psi) + psi"],
            index=0,
            help="Choose how the two phase fields are merged into one colour map."
        )
    with col_b:
        show_potential = st.checkbox("Overlay electric-potential proxy (-α·c)", value=True)

    # ------------------------------------------------------------------
    #  Build the fields for the selected frame
    # ------------------------------------------------------------------
    t, phi_view, c_view, psi_view = snapshots[frame_idx]

    material = build_material(phi_view, psi_view, material_method)

    # electric-potential proxy: the only term that couples to c is -α·c
    potential = -alpha * c_view

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------
    if mode.startswith("2D"):
        fig_mat, ax_mat = plt.subplots(figsize=(6, 5))
        # discrete colour map for 0/1/2 (electrolyte / Ag / Cu)
        if material_method in ["phi + 2*psi (simple)",
                               "phi*(1-psi) + 2*psi",
                               "max(phi, psi) + psi"]:
            cmap_mat = plt.cm.get_cmap("Set1", 3)      # 3 discrete colours
            im_mat = ax_mat.imshow(material.T,
                                   origin='lower',
                                   extent=[0, 1, 0, 1],
                                   cmap=cmap_mat,
                                   vmin=0, vmax=2)
            cbar_mat = plt.colorbar(im_mat, ax=ax_mat, ticks=[0, 1, 2])
            cbar_mat.ax.set_yticklabels(['electrolyte', 'Ag shell', 'Cu core'])
        else:                                          # continuous (e.g. h·(φ²+ψ²))
            im_mat = ax_mat.imshow(material.T,
                                   origin='lower',
                                   extent=[0, 1, 0, 1],
                                   cmap=cmap_choice)
            plt.colorbar(im_mat, ax=ax_mat, label="h·(φ²+ψ²)")

        ax_mat.set_title(f"Material @ t* = {t:.5f}")
        st.pyplot(fig_mat)

        # ---- potential overlay (optional) ----
        if show_potential:
            fig_pot, ax_pot = plt.subplots(figsize=(6, 5))
            im_pot = ax_pot.imshow(potential.T,
                                   origin='lower',
                                   extent=[0, 1, 0, 1],
                                   cmap="RdBu_r")
            plt.colorbar(im_pot, ax=ax_pot, label="Potential proxy  -α·c")
            ax_pot.set_title(f"Electric-potential proxy @ t* = {t:.5f}")
            st.pyplot(fig_pot)

        # ---- combined view (material + contour of potential) ----
        if show_potential:
            fig_comb, ax_comb = plt.subplots(figsize=(6, 5))
            # material background (discrete)
            if material_method in ["phi + 2*psi (simple)",
                                   "phi*(1-psi) + 2*psi",
                                   "max(phi, psi) + psi"]:
                ax_comb.imshow(material.T,
                               origin='lower',
                               extent=[0, 1, 0, 1],
                               cmap=cmap_mat,
                               vmin=0, vmax=2, alpha=0.7)
            else:
                ax_comb.imshow(material.T,
                               origin='lower',
                               extent=[0, 1, 0, 1],
                               cmap=cmap_choice, alpha=0.7)

            # potential contours
            cs = ax_comb.contour(potential.T,
                                 levels=12,
                                 cmap="plasma",
                                 linewidths=0.8,
                                 alpha=0.9)
            ax_comb.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
            ax_comb.set_title("Material + Potential contours")
            st.pyplot(fig_comb)

    else:   # ----------------------------------------------------------- 3-D
        # For 3-D we show three orthogonal slices (same as the original UI)
        cx = phi_view.shape[0] // 2
        cy = phi_view.shape[1] // 2
        cz = phi_view.shape[2] // 2

        fig_mat, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sl, label in zip(axes,
                                 [material[cx, :, :], material[:, cy, :], material[:, :, cz]],
                                 ["x-slice", "y-slice", "z-slice"]):
            if material_method in ["phi + 2*psi (simple)",
                                   "phi*(1-psi) + 2*psi",
                                   "max(phi, psi) + psi"]:
                im = ax.imshow(sl.T, origin='lower', cmap=cmap_mat, vmin=0, vmax=2)
            else:
                im = ax.imshow(sl.T, origin='lower', cmap=cmap_choice)
            ax.set_title(label)
            ax.axis('off')
        fig_mat.suptitle(f"Material (3-D slices) @ t* = {t:.5f}")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, sl, label in zip(axes,
                                     [potential[cx, :, :], potential[:, cy, :], potential[:, :, cz]],
                                     ["x-slice", "y-slice", "z-slice"]):
                im = ax.imshow(sl.T, origin='lower', cmap="RdBu_r")
                ax.set_title(label)
                ax.axis('off')
            fig_pot.suptitle(f"Potential proxy (-α·c) @ t* = {t:.5f}")
            plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, label="-α·c")
            st.pyplot(fig_pot)
